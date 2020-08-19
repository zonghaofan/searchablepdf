# coding: utf8
"""根据 tif 和 相应的 xml 生成双层 PDF。
"""
import argparse
import json
import logging
import os
import shutil
import time
from datetime import datetime
from os.path import getsize as gs
from pathlib import Path

import numpy as np
from lxml.etree import fromstring
from PIL import Image
import cv2

import xmljson
from reportlab import platypus
from reportlab.lib.pagesizes import A3, A4, LETTER
from reportlab.lib.units import cm, inch, mm, pica
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Image as Rmage
from reportlab.platypus import SimpleDocTemplate

logger = logging.getLogger(__name__)


def tif2jpg(
    tif: str,
    jpg: str,
    min_size: float,
    max_size: float,
    init_quality: int,
    step: int,
    base_dir: str,
) -> None:
    if not base_dir.endswith("/"):
        base_dir += "/"
    tif = Path(tif)
    tif_parent = str(tif.parent) + "/"  # Path.parent 最后不加 /，而我们需要
    tif_name = tif.stem
    img = Image.open(str(tif))

    if not Path(jpg).parent.exists():
        logger.warning(f"目录 {Path(jpg).parent} 不存在，将被创建。")
        Path(jpg).parent.mkdir(parents=True)
    quality = init_quality
    img.convert("L").save(jpg, quality=quality, dpi=(300.0, 300.0))
    cur = quality
    gap = step
    op_cnt = 0
    while (gs(jpg) * 1.0 / 1024 / 1024) < min_size:  # unit: metabytes
        cur += gap
        img.convert("L").save(jpg, quality=cur, dpi=(300.0, 300.0))
        op_cnt += 1
    while (gs(jpg) * 1.0 / 1024 / 1024) > max_size:  # unit: metabytes
        cur -= gap
        img.convert("L").save(jpg, quality=cur, dpi=(300.0, 300.0))
        op_cnt += 1
    logger.debug(
        f"{tif} -> {jpg}, 调整次数 {op_cnt}, 最终质量 {cur}, 最终大小 {gs(jpg) * 1.0 / 1024 / 1024:.2f}MB"
    )


def xml2dict(xml_file, raw_json=None, format_json=None):
    content = Path(xml_file).read_text(encoding="utf8")
    # 如果包含 xml 声明解析会报错，此处去掉
    if content.splitlines()[0].startswith("<?xml version="):
        content = "\n".join(content.splitlines()[1:])
    data = xmljson.badgerfish.data(fromstring(content))
    result = {}
    zhengwens = data["root"]["版元数据"]["正文组"]["正文"]
    if not isinstance(zhengwens, list):
        zhengwens = [zhengwens]
    for i in zhengwens:
        pianmu = i["@篇目号"]
        yinti = i["引题"]["$"] if i["引题"] else None
        biaoti = i["标题"]["$"] if i["标题"] else None
        futi = i["副题"]["$"] if i["副题"] else None
        xiaobiaoti = i["小标题"]["$"] if i["小标题"] else None
        lanmu = i["栏目"]["$"] if i["栏目"] else None
        author = i["作者"]["$"] if i["作者"] else None
        zhengwen_coords = data["root"]["字符坐标"]["正文组"]["正文"]
        if not isinstance(zhengwen_coords, list):
            zhengwen_coords = [zhengwen_coords]
        for j in zhengwen_coords:
            if j["@篇目号"] == pianmu:
                yinti_coord = (
                    [
                        list(map(int, item.split(",")))
                        for item in j["引题"]["$"].split(";")
                    ]
                    if j["引题"]
                    else None
                )
                biaoti_coord = (
                    [
                        list(map(int, item.split(",")))
                        for item in j["标题"]["$"].split(";")
                    ]
                    if j["标题"]
                    else None
                )
                futi_coord = (
                    [
                        list(map(int, item.split(",")))
                        for item in j["副题"]["$"].split(";")
                    ]
                    if j["副题"]
                    else None
                )
                xiaobiaoti_coord = (
                    [
                        list(map(int, item.split(",")))
                        for item in j["小标题"]["$"].split(";")
                    ]
                    if j["小标题"]
                    else None
                )
                lanmu_coord = (
                    [
                        list(map(int, item.split(",")))
                        for item in j["栏目"]["$"].split(";")
                    ]
                    if j["栏目"]
                    else None
                )
                author_coord = (
                    [
                        list(map(int, item.split(",")))
                        for item in j["作者"]["$"].split(";")
                    ]
                    if j["作者"]
                    else None
                )
                break
        result[pianmu] = {
            "引题": yinti,
            "标题": biaoti,
            "副题": futi,
            "小标题": xiaobiaoti,
            "栏目": lanmu,
            "作者": author,
            "引题坐标": yinti_coord,
            "标题坐标": biaoti_coord,
            "副题坐标": futi_coord,
            "小标题坐标": xiaobiaoti_coord,
            "栏目坐标": lanmu_coord,
            "作者坐标": author_coord,
        }
    if '图片组' in data["root"]["版元数据"]:
        images = data["root"]["版元数据"]["图片组"]["图片"]
        if not isinstance(images, list):
            images = [images]
        for i in images:
            pianmu = i["@篇目号"]
            yinti = i["图片引题"]["$"] if i["图片引题"] else None
            biaoti = i["图片标题"]["$"] if i["图片标题"] else None
            futi = i["图片副题"]["$"] if i["图片副题"] else None
            lanmu = i["栏目"]["$"] if i["栏目"] else None
            author = i["图片作者"]["$"] if i["图片作者"] else None
            image_coords = data["root"]["字符坐标"]["图片组"]["图片"]
            if not isinstance(image_coords, list):
                image_coords = [image_coords]
            for j in image_coords:
                if j["@篇目号"] == pianmu:
                    yinti_coord = (
                        [
                            list(map(int, item.split(",")))
                            for item in j["图片引题"]["$"].split(";")
                        ]
                        if j["图片引题"]
                        else None
                    )
                    biaoti_coord = (
                        [
                            list(map(int, item.split(",")))
                            for item in j["图片标题"]["$"].split(";")
                        ]
                        if j["图片标题"]
                        else None
                    )
                    futi_coord = (
                        [
                            list(map(int, item.split(",")))
                            for item in j["图片副题"]["$"].split(";")
                        ]
                        if j["图片副题"]
                        else None
                    )
                    lanmu_coord = (
                        [
                            list(map(int, item.split(",")))
                            for item in j["栏目"]["$"].split(";")
                        ]
                        if j["栏目"]
                        else None
                    )
                    author_coord = (
                        [
                            list(map(int, item.split(",")))
                            for item in j["图片作者"]["$"].split(";")
                        ]
                        if j["图片作者"]
                        else None
                    )
                    break
            result[f"img_{pianmu}"] = {
                "引题": yinti,
                "标题": biaoti,
                "副题": futi,
                "栏目": lanmu,
                "作者": author,
                "引题坐标": yinti_coord,
                "标题坐标": biaoti_coord,
                "副题坐标": futi_coord,
                "栏目坐标": lanmu_coord,
                "作者坐标": author_coord,
            }
    if raw_json:  # xml 转成 dict 后的原始结果
        with open(raw_json, "w", encoding="utf8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    if format_json:  # 格式化后的，key 为篇目号，为方便后期生成 PDF
        with open(format_json, "w", encoding="utf8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
    return result


def generate_searchable_pdf(
    image_file,
    xml_file,
    font_name=None,
    font_path=None,
    outpdf=None,
    raw_json=None,
    format_json=None,
    plot_rec=False,
    rec_file='./cv_image_rect.jpg'):
    """针对单张图片的双层 PDF 生成。
    """
    image = Image.open(image_file)
    if plot_rec:
        cv_image = cv2.imread(image_file)
    #图片是300 dpi, pdf 是 72 dpi
    ratio = 300 / 72
    pagesize = (image.width / ratio, image.height / ratio)
    # pagesize = (29.207 * cm, 40.212 * cm)
    # pagesize = LETTER
    logger.debug(f"image info={image.info}")
    logger.debug(f"pagesize={pagesize}")
    # pagesize = (3450, 4750)
    # Use Canvas to generate pdf
    if not outpdf:
        outpdf = f"{Path(image_file).stem}.pdf"
        if Path(outpdf).exists():
            outpdf = f"{Path(image_file).stem}_{now}.pdf"
    if not Path(outpdf).parent.exists():
        logger.warning(f"目录 {Path(outpdf).parent} 不存在，将被创建。")
        Path(outpdf).parent.mkdir(parents=True)
    c = canvas.Canvas(outpdf, pagesize=pagesize)
    y_offset = 5

    strings_and_xys = xml2dict(xml_file, raw_json=raw_json, format_json=format_json)
    print('===strings_and_xys:', strings_and_xys)

    def calculate_fontsize(xy):
        area = abs(xy[2] - xy[0]) * abs(xy[3] - xy[1])
        diagonal = np.sqrt((xy[2] - xy[0]) ** 2 + (xy[3] - xy[1]) ** 2)
        size_pre = 0.6701827331976165 * diagonal - 9.424118771648466
        # size = 0.6701827331976165 * diagonal - 9.424118771648466
        size = 0.5609483475409048 * diagonal + 1.956936059970495
        return diagonal, area, size_pre, size

    for k, v in strings_and_xys.items():
        logger.debug(f" pianmu {k} ".center(100, "="))
        pianmu = v
        xys = pianmu["引题坐标"]
        chars = str(pianmu["引题"]).replace(" ", "") if pianmu["引题"] else None
        if xys and chars:
            for xml_xy, char in zip(xys, chars):
                if plot_rec:
                    cv2.rectangle(
                        cv_image,
                        (xml_xy[0], xml_xy[1]),
                        (xml_xy[2], xml_xy[3]),
                        thickness=5,
                        color=(255, 0, 0),
                    )
                xml_xy = [i / ratio for i in xml_xy]
                diagonal, area, fontsize_pre, fontsize = calculate_fontsize(xml_xy)
                logger.debug(
                    f"char={char}, diagonal={diagonal}, area={area}, fontsize={fontsize}, pre fontsize={fontsize_pre}"
                )
                # fontsize *= ratio
                c.setFont(font_name, size=fontsize)
                c.drawString(xml_xy[0], (pagesize[1] - xml_xy[3]) + y_offset, char)
        xys = pianmu["标题坐标"]
        chars = str(pianmu["标题"]).replace(" ", "") if pianmu["标题"] else None
        if xys and chars:
            for xml_xy, char in zip(xys, chars):
                if plot_rec:
                    cv2.rectangle(
                        cv_image,
                        (xml_xy[0], xml_xy[1]),
                        (xml_xy[2], xml_xy[3]),
                        thickness=5,
                        color=(255, 0, 0),
                    )
                xml_xy = [i / ratio for i in xml_xy]
                diagonal, area, fontsize_pre, fontsize = calculate_fontsize(xml_xy)
                logger.debug(
                    f"char={char}, diagonal={diagonal}, area={area}, fontsize={fontsize}, pre fontsize={fontsize_pre}"
                )
                c.setFont(font_name, size=fontsize)
                c.drawString(xml_xy[0], (pagesize[1] - xml_xy[3]) + y_offset, char)
        xys = pianmu["副题坐标"]
        chars = str(pianmu["副题"]).replace(" ", "") if pianmu["副题"] else None
        if xys and chars:
            for xml_xy, char in zip(xys, chars):
                if plot_rec:
                    cv2.rectangle(
                        cv_image,
                        (xml_xy[0], xml_xy[1]),
                        (xml_xy[2], xml_xy[3]),
                        thickness=5,
                        color=(255, 0, 0),
                    )
                xml_xy = [i / ratio for i in xml_xy]
                diagonal, area, fontsize_pre, fontsize = calculate_fontsize(xml_xy)
                logger.debug(
                    f"char={char}, diagonal={diagonal}, area={area}, fontsize={fontsize}, pre fontsize={fontsize_pre}"
                )
                # fontsize = 20
                c.setFont(font_name, size=fontsize)
                c.drawString(xml_xy[0], (pagesize[1] - xml_xy[3]) + y_offset, char)
        xys = pianmu.get('小标题坐标', None)
        chars = str(pianmu["小标题"]).replace(" ", "") if pianmu.get("小标题", None) else None
        if xys and chars:
            for xml_xy, char in zip(xys, chars):
                if plot_rec:
                    cv2.rectangle(
                        cv_image,
                        (xml_xy[0], xml_xy[1]),
                        (xml_xy[2], xml_xy[3]),
                        thickness=5,
                        color=(255, 0, 0),
                    )

                xml_xy = [i / ratio for i in xml_xy]
                diagonal, area, fontsize_pre, fontsize = calculate_fontsize(xml_xy)
                logger.debug(
                    f"char={char}, diagonal={diagonal}, area={area}, fontsize={fontsize}, pre fontsize={fontsize_pre}"
                )
                c.setFont(font_name, size=fontsize)
                c.drawString(xml_xy[0], (pagesize[1] - xml_xy[3]) + y_offset, char)
        xys = pianmu["栏目坐标"]
        chars = str(pianmu["栏目"]).replace(" ", "") if pianmu["栏目"] else None
        if xys and chars:
            for xml_xy, char in zip(xys, chars):
                if plot_rec:
                    cv2.rectangle(
                        cv_image,
                        (xml_xy[0], xml_xy[1]),
                        (xml_xy[2], xml_xy[3]),
                        thickness=5,
                        color=(255, 0, 0),
                    )
                xml_xy = [i / ratio for i in xml_xy]
                diagonal, area, fontsize_pre, fontsize = calculate_fontsize(xml_xy)
                logger.debug(
                    f"char={char}, diagonal={diagonal}, area={area}, fontsize={fontsize}, pre fontsize={fontsize_pre}"
                )
                c.setFont(font_name, size=fontsize)
                c.drawString(xml_xy[0], (pagesize[1] - xml_xy[3]) + y_offset, char)
        xys = pianmu["作者坐标"]
        chars = str(pianmu["作者"]).replace(" ", "") if pianmu["作者"] else None
        if xys and chars:
            for xml_xy, char in zip(xys, chars):
                if plot_rec:
                    cv2.rectangle(
                        cv_image,
                        (xml_xy[0], xml_xy[1]),
                        (xml_xy[2], xml_xy[3]),
                        thickness=5,
                        color=(255, 0, 0),
                    )
                xml_xy = [i / ratio for i in xml_xy]
                diagonal, area, fontsize_pre, fontsize = calculate_fontsize(xml_xy)
                logger.debug(
                    f"char={char}, diagonal={diagonal}, area={area}, fontsize={fontsize}, pre fontsize={fontsize_pre}"
                )
                c.setFont(font_name, size=fontsize)
                c.drawString(xml_xy[0], (pagesize[1] - xml_xy[3]) + y_offset, char)

    c.drawImage(
        image_file, 0, 0, width=image.width / ratio, height=image.height / ratio
    )
    c.showPage()
    c.save()
    if plot_rec:
        if not rec_file:
            raise ValueError(" plot_rec 为 True 时必须指定 rec_file。")
        cv2.imwrite(rec_file, cv_image)


def debug_main():
    import glob
    # tif_dir = args.tif_dir
    # xml_dir = args.xml_dir
    # jpg_dir = args.jpg_dir
    # pdf_dir = args.pdf_dir
    #debug
    tif_dir = "/data/newspaper/guotu-searchablepdf/src/test/data/tif"
    xml_dir = "/data/newspaper/guotu-searchablepdf/src/test/data/xml"
    jpg_dir = "/data/newspaper/guotu-searchablepdf/src/test/data/jpg"
    pdf_dir = "/data/newspaper/guotu-searchablepdf/src/test/data/pdf"

    jpg_min_size = 0.95
    jpg_max_size = 1.0
    jpg_init_quality = 15
    jpg_adjust_step = 3
    fontname = 'simsun'
    fontpath = '/usr/share/fonts/simsun.ttf'
    pdfmetrics.registerFont(TTFont(fontname, fontpath))
    tifs = glob.glob(tif_dir+"/*.tif")
    total = len(tifs)
    data = {}
    print('==tifs:', tifs)
    for i, tif in enumerate(tifs):
        # if i<1:
            xml =  tif.replace('tif', 'xml')
            print('===xml:', xml)
            if not os.path.exists(xml):
                logger.error(f"找不到 {tif} 对应的 xml 文件 {xml}。")
                data[str(tif)] = {"xml": str(xml), "pdf": None}
                continue
            jpg = tif.replace('.tif', '.jpg')
            print('===jpg:', jpg)

            pdf = tif.replace('.tif', '.pdf')
            print('===pdf:', pdf)
            try:
                logger.info(f"[{i+1}/{total}] 正在根据 {tif} 和 {xml} 生成 {pdf} ...")
                start = time.time()
                jpg_start = time.time()
                tif2jpg(
                    str(tif),
                    str(jpg),
                    min_size=jpg_min_size,
                    max_size=jpg_max_size,
                    init_quality=jpg_init_quality,
                    step=jpg_adjust_step,
                    base_dir=tif_dir,
                )
                jpg_cost = time.time() - jpg_start
                pdf_start = time.time()
                generate_searchable_pdf(
                    jpg, xml, fontname, fontpath, outpdf=pdf, plot_rec=True, raw_json=xml.replace('.xml','_raw.json'), format_json=xml.replace('.xml','_ref.json')
                )
                pdf_cost = time.time() - pdf_start
                cost = time.time() - start
                data[str(tif)] = {"xml": str(xml), "pdf": str(pdf)}
                logger.info(
                    f"完成，总计耗时 {cost:.4f}s，其中 TIF 转 JPG 耗时 {jpg_cost:.4f}s，JPG 和 XML 生成 PDF 耗时 {pdf_cost:.4f}s。"
                )
            except Exception as e:
                logger.error("生成时发生错误。", exc_info=True)
                data[str(tif)] = {"xml": str(xml), "pdf": None}


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="""TIF 转 PDF 并保持源目录结构，使用时请将本程序和 tif 目录、xml 目录放在同一目录下。"""
    # )
    # parser.add_argument("--tif_dir", help="tif 文件目录，默认为 tif/", default="tif/")
    # parser.add_argument("--jpg_dir", help="jpg 文件目录，默认为 jpg/", default="jpg/")
    # parser.add_argument("--xml_dir", help="xml 文件目录，默认为 xml/", default="xml/")
    # parser.add_argument("--pdf_dir", help="pdf 文件目录，默认为 pdf/", default="pdf/")
    # parser.add_argument(
    #     "--json",
    #     help="是否保存中间过程产生的 json，默认否。如果是，则默认保存在 json/ 目录，如需改变目录需使用 --json_dir 指定",
    #     default=False,
    #     action="store_true",
    # )
    # parser.add_argument("--json_dir", help="json 文件目录，默认为 json/", default="json/")
    # parser.add_argument("--min_size", type=float, help="输出的 jpg 的最小尺寸(MB)", default="0.95")
    # parser.add_argument("--max_size", type=float, help="输出的 jpg 的最大尺寸(MB)", default="1.0")
    # parser.add_argument(
    #     "--init_quality", type=int, help="jpg的初始质量, 这个值越近最终质量, 调整次数越少", default="15"
    # )
    # parser.add_argument(
    #     "--step", type=int, help="jpg的质量调整步长, 越大越快, 过大可能无法调整到目标值", default="3"
    # )
    # args = parser.parse_args()
    debug_main()
