from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile
import pytest
from tabpollution import intake
NS='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
REL='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PKG='http://schemas.openxmlformats.org/package/2006/relationships'
def make_xlsx(target='worksheets/sheet1.xml'):
 files={'xl/workbook.xml':f'<workbook xmlns="{NS}" xmlns:r="{REL}"><sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>','xl/_rels/workbook.xml.rels':f'<Relationships xmlns="{PKG}"><Relationship Id="rId1" Type="worksheet" Target="{target}"/></Relationships>','xl/worksheets/sheet1.xml':f'<worksheet xmlns="{NS}"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>X1</t></is></c><c r="B1" t="inlineStr"><is><t>X2</t></is></c><c r="C1" t="inlineStr"><is><t>Y1</t></is></c><c r="D1" t="inlineStr"><is><t>Y2</t></is></c></row><row r="2"><c r="A2"><v>1</v></c><c r="B2"><v>2</v></c><c r="C2"><v>3</v></c><c r="D2"><v>9</v></c></row><row r="3"/><row r="4"><c r="A4"><v>4</v></c><c r="B4"><v>5</v></c><c r="C4"><v>6</v></c><c r="D4"><v>9</v></c></row></sheetData></worksheet>'}
 out=BytesIO()
 with ZipFile(out,'w',ZIP_DEFLATED) as z:
  for n,v in files.items(): z.writestr(n,v)
 return out.getvalue()
def record():
 return {'source_format':'excel','source_column_order':['X1','X2','Y1','Y2'],'features':['X1','X2'],'target':'Y1','exclude_columns':['Y2'],'header':0,'sheet_name':'Sheet1'}
def test_xlsx_fallback_reads_numeric_rows_and_maps_target(monkeypatch):
 def missing(*a,**k): raise ImportError('openpyxl missing')
 monkeypatch.setattr(intake.pd,'read_excel',missing)
 frame=intake._parse_payload(record(),make_xlsx())
 assert list(frame.columns)==['X1','X2','Y1']
 assert frame.values.tolist()==[['1','2','3'],['4','5','6']]
def test_xlsx_rejects_escaping_sheet_relationship(monkeypatch):
 monkeypatch.setattr(intake.pd,'read_excel',lambda *a,**k: (_ for _ in ()).throw(ImportError()))
 with pytest.raises(intake.IntakeError,match='escapes workbook'): intake._parse_payload(record(),make_xlsx('../../outside.xml'))
def test_column_alias_collision_is_rejected():
 row={'source_format':'csv','source_column_order':['raw_x','x'],'source_column_renames':{'raw_x':'x'},'features':['x'],'target':'y','exclude_columns':[]}
 with pytest.raises(intake.IntakeError,match='not one-to-one'): intake._parse_payload(row,b'raw_x,x\n1,2\n')
