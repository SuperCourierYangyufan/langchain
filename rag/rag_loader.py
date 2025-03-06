from langchain_community.document_loaders import TextLoader, PDFMinerLoader, \
  UnstructuredExcelLoader, DirectoryLoader, BiliBiliLoader
from onnxruntime.transformers.shape_infer_helper import file_path

# loader = TextLoader(file_path="E:\\down\\新建文本文档.txt")
# print(loader.load())

pdf_loader = PDFMinerLoader  (
  file_path="C:\\Users\\Administrator\\Desktop\\规则引擎开发工作量清单.pdf")
print(pdf_loader.load_and_split())


excel_loader = UnstructuredExcelLoader(
  file_path="C:\\Users\\Administrator\\Desktop\\规则引擎开发工作量清单.xlsx")
print(excel_loader.load())
#
# dire_loader = DirectoryLoader(path = "C:\\Users\\Administrator\\Desktop",glob="*.xlsx")
# docs = dire_loader.load()
# print(len(docs))
# print(docs)
#
#
# bilibili_loader = BiliBiliLoader(video_urls=[
#   'https://www.bilibili.com/video/BV1aNNGedEoH?spm_id_from=333.788.player.switch&vd_source=77add9524f1b913462be3779544b1817&p=31'],
#     sessdata="buvid3=04DB488B-DB98-8CE3-8713-E624EF25D58269042infoc; b_nut=1740485069; _uuid=1982441E-DD81-D118-10FD2-D794994AF58D50865infoc; enable_web_push=DISABLE; enable_feed_channel=DISABLE; buvid4=79635C1D-6175-9407-967B-4BF60A7EFE4A69628-025022512-MBQOq%2FL7TPPCgmsm8RQeuA%3D%3D; buvid_fp=df28f8ce6bedf5dff5950b33aad48dc4; header_theme_version=CLOSE; DedeUserID=27260986; DedeUserID__ckMd5=e1497ef2fbeca085; rpdid=|(um~k)l)uJk0J'u~R|RmRYJ|; CURRENT_QUALITY=80; hit-dyn-v2=1; LIVE_BUVID=AUTO1017406522348210; PVID=12; bili_ticket=eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NDEzNDQ5MjQsImlhdCI6MTc0MTA4NTY2NCwicGx0IjotMX0.SSd5JOGCbN1UbufvCbckTpNbd6sWzFqZVEjSZOpbwx8; bili_ticket_expires=1741344864; SESSDATA=870e7741%2C1756637724%2C7283b%2A32CjCrGr3--lR2KNyjsfA4Kfda3z73TsDQV51CnT_MSjs_Bk1amLKh_dzcrrNIvL3Z_8ESVjMzckRwV3V5cGtKYXlTa2JIV0VXRzAyM1cyWTBJVDFqRFZ3WDgzVVdsR2N5MGlER1pIeHFrZjZVS2NKdFJweU4wd0tkMms4cnJweC1lLVFOV0VhRWVRIIEC; bili_jct=e27d9603920a4e7649b8379ec4a40265; sid=8pg02qoi; CURRENT_FNVAL=4048; bp_t_offset_27260986=1041203564082364416; b_lsid=19F10F52D_1956B66A0AE; bsource=search_baidu; home_feed_column=4; browser_resolution=1125-889")
# print(bilibili_loader.load())
#
