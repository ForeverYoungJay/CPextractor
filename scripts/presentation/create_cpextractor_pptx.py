from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


SLIDES = [
    {
        "title": "CPextractor",
        "subtitle": "从晶体塑性论文到可追溯、可检索、可复核的参数数据库",
        "bullets": [
            "Extraction, Grounding, Committee, Confidence, Database",
            "Evidence-grounded literature-to-database pipeline",
        ],
    },
    {
        "title": "项目要解决的问题",
        "bullets": [
            "晶体塑性参数散落在正文、表格、补充材料和引用文献里",
            "科研复用不仅需要抽取数值，还需要单位归一、作用范围、来源与证据",
            "手工整理难以规模化支撑数据库建设、检索与 RAG 问答",
        ],
    },
    {
        "title": "CPextractor 总览",
        "bullets": [
            "输入：Scopus 检索、DOI 列表、本地全文目录",
            "解析：Elsevier XML 转为 sections、tables、references",
            "抽取：两阶段 LLM，先选文件，再做 schema-constrained extraction",
            "后处理：参数归一化、材料/相/作用范围绑定、evidence grounding",
            "输出：parameter_claims、审查材料、PostgreSQL、向量检索与聊天问答",
        ],
    },
    {
        "title": "端到端流水线",
        "bullets": [
            "获取文献并建立本地全文资产",
            "解析正文、表格和引用关系",
            "抽取候选参数记录到 parameters.registry",
            "进行确定性后处理与证据回链",
            "运行 LLM committee 与 meta judge",
            "融合规则分和 judge 分数，决定置信度、质量层级和是否入库",
        ],
    },
    {
        "title": "质量控制栈",
        "bullets": [
            "Layer 1: Rule validation",
            "Layer 2: Evidence grounding",
            "Layer 3: Multi-agent LLM evaluation",
            "Layer 4: Confidence fusion and quality tiering",
            "低质量论文可以被 gate 掉，但完整审计痕迹仍然保留",
        ],
    },
    {
        "title": "有没有 Rule-Based Judge",
        "bullets": [
            "没有一个单独命名为 rule-based judge 的 agent",
            "但有一层确定性规则校验，功能上相当于 rule-based QA / validator",
            "典型检查：evidence 缺失、关键参数缺值、单位缺失、明显负值、scope 绑定不一致",
            "规则结果形成 quality_checks 报告和 rule_score",
            "这些结果会和 LLM committee 输出一起进入最终 confidence fusion",
        ],
    },
    {
        "title": "LLM Committee 怎么工作",
        "bullets": [
            "Evidence judge：判断参数是否被当前证据支持",
            "Normalization judge：检查 canonical mapping、unit、SI conversion",
            "Consistency judge：检查跨参数、跨材料、binding、model scope 的一致性",
            "Meta judge：综合 committee、rule report、evidence report，给出文档级 verdict",
            "参数级三票先做 consensus，再做 policy adjustment，最后形成 review_required",
        ],
    },
    {
        "title": "Rule + Committee 融合逻辑",
        "bullets": [
            "参数级分数同时考虑 rule penalty、LLM audit score、基础证据质量",
            "文档级分数融合 rule_score 与 overall_score",
            "对低风险 table grounding 分歧放宽处罚",
            "对实际上 SI 换算正确的误报做 suppress",
            "输出 document confidence、parameter confidence、quality tier、review escalation",
        ],
    },
    {
        "title": "数据模型与数据库",
        "bullets": [
            "中间结构：parameters.registry",
            "最小可信单元：parameter_claims",
            "证据对象：evidence_objects",
            "最终层级：materials -> phases -> conditions -> models -> claims",
            "数据库包含结构化表、引用表、向量表和 evaluation tables",
        ],
    },
    {
        "title": "评估与应用场景",
        "bullets": [
            "Extraction correctness：字段、数值、单位、引用信息",
            "Judge correctness：和 reviewed queue 对比，评估准确率与 calibration",
            "Database utility：检索命中率、问答 grounding、分析任务成功率",
            "应用方向：参数数据库、证据约束 RAG、材料与机制分析",
        ],
    },
    {
        "title": "推荐演示路线",
        "bullets": [
            "选一篇 DOI",
            "展示 parse 后的 sections / tables / references",
            "展示 extractor 输出与 postprocess 结果",
            "展示 evidence grounding、committee audit、confidence fusion",
            "展示最终入库、检索与聊天问答",
        ],
    },
    {
        "title": "收尾",
        "bullets": [
            "CPextractor 不是单一抽取器，而是一条 evidence-grounded curation pipeline",
            "核心价值是把抽取、规范化、审计、置信度建模和数据库落地放进同一系统",
            "适合两个论文方向：scientific IE + evidence-grounded database system",
        ],
    },
]


EMU_PER_INCH = 914400
SLIDE_W = int(13.333 * EMU_PER_INCH)
SLIDE_H = int(7.5 * EMU_PER_INCH)


def xml_header() -> str:
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'


def content_types(slide_count: int) -> str:
    overrides = "\n".join(
        [
            '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
            '<Override PartName="/ppt/presProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"/>',
            '<Override PartName="/ppt/viewProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"/>',
            '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
            '<Override PartName="/ppt/tableStyles.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.tableStyles+xml"/>',
            '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
            '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
            '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
            '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
        ]
        + [
            f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
            for i in range(1, slide_count + 1)
        ]
    )
    return f"""{xml_header()}
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  {overrides}
</Types>"""


def root_rels() -> str:
    return f"""{xml_header()}
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""


def app_xml(slide_count: int) -> str:
    return f"""{xml_header()}
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Office PowerPoint</Application>
  <PresentationFormat>On-screen Show (16:9)</PresentationFormat>
  <Slides>{slide_count}</Slides>
  <Notes>0</Notes>
  <HiddenSlides>0</HiddenSlides>
  <MMClips>0</MMClips>
  <ScaleCrop>false</ScaleCrop>
  <HeadingPairs>
    <vt:vector size="2" baseType="variant">
      <vt:variant><vt:lpstr>Theme</vt:lpstr></vt:variant>
      <vt:variant><vt:i4>1</vt:i4></vt:variant>
    </vt:vector>
  </HeadingPairs>
  <TitlesOfParts>
    <vt:vector size="1" baseType="lpstr">
      <vt:lpstr>Office Theme</vt:lpstr>
    </vt:vector>
  </TitlesOfParts>
  <Company>OpenAI Codex</Company>
  <LinksUpToDate>false</LinksUpToDate>
  <SharedDoc>false</SharedDoc>
  <HyperlinksChanged>false</HyperlinksChanged>
  <AppVersion>16.0000</AppVersion>
</Properties>"""


def core_xml() -> str:
    return f"""{xml_header()}
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>CPextractor Project Overview</dc:title>
  <dc:creator>OpenAI Codex</dc:creator>
  <cp:lastModifiedBy>OpenAI Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">2026-04-09T00:00:00Z</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">2026-04-09T00:00:00Z</dcterms:modified>
</cp:coreProperties>"""


def presentation_xml(slide_count: int) -> str:
    slide_ids = "\n".join(
        f'    <p:sldId id="{255 + i}" r:id="rId{i + 1}"/>'
        for i in range(1, slide_count + 1)
    )
    return f"""{xml_header()}
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
 saveSubsetFonts="1" autoCompressPictures="0">
  <p:sldMasterIdLst>
    <p:sldMasterId id="2147483648" r:id="rId{slide_count + 1}"/>
  </p:sldMasterIdLst>
  <p:sldIdLst>
{slide_ids}
  </p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}"/>
  <p:notesSz cx="6858000" cy="9144000"/>
  <p:defaultTextStyle>
    <a:defPPr/>
    <a:lvl1pPr marL="0" indent="0"/>
    <a:lvl2pPr marL="457200" indent="0"/>
    <a:lvl3pPr marL="914400" indent="0"/>
  </p:defaultTextStyle>
</p:presentation>"""


def presentation_rels(slide_count: int) -> str:
    relationships = "\n".join(
        f'  <Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        for i in range(1, slide_count + 1)
    )
    return f"""{xml_header()}
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{relationships}
  <Relationship Id="rId{slide_count + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>
  <Relationship Id="rId{slide_count + 2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/presProps" Target="presProps.xml"/>
  <Relationship Id="rId{slide_count + 3}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/viewProps" Target="viewProps.xml"/>
  <Relationship Id="rId{slide_count + 4}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>
  <Relationship Id="rId{slide_count + 5}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/tableStyles" Target="tableStyles.xml"/>
</Relationships>"""


def theme_xml() -> str:
    return f"""{xml_header()}
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="CPextractor Theme">
  <a:themeElements>
    <a:clrScheme name="CPextractor Colors">
      <a:dk1><a:srgbClr val="111827"/></a:dk1>
      <a:lt1><a:srgbClr val="F7F7F5"/></a:lt1>
      <a:dk2><a:srgbClr val="1F2937"/></a:dk2>
      <a:lt2><a:srgbClr val="FFFDF8"/></a:lt2>
      <a:accent1><a:srgbClr val="0F766E"/></a:accent1>
      <a:accent2><a:srgbClr val="C2410C"/></a:accent2>
      <a:accent3><a:srgbClr val="2563EB"/></a:accent3>
      <a:accent4><a:srgbClr val="7C3AED"/></a:accent4>
      <a:accent5><a:srgbClr val="4D7C0F"/></a:accent5>
      <a:accent6><a:srgbClr val="B91C1C"/></a:accent6>
      <a:hlink><a:srgbClr val="2563EB"/></a:hlink>
      <a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="CPextractor Fonts">
      <a:majorFont>
        <a:latin typeface="Aptos Display"/>
        <a:ea typeface="Microsoft YaHei"/>
        <a:cs typeface="Arial"/>
      </a:majorFont>
      <a:minorFont>
        <a:latin typeface="Aptos"/>
        <a:ea typeface="Microsoft YaHei"/>
        <a:cs typeface="Arial"/>
      </a:minorFont>
    </a:fontScheme>
    <a:fmtScheme name="CPextractor Formats">
      <a:fillStyleLst>
        <a:solidFill><a:schemeClr val="lt1"/></a:solidFill>
        <a:solidFill><a:schemeClr val="accent1"/></a:solidFill>
        <a:solidFill><a:schemeClr val="accent2"/></a:solidFill>
      </a:fillStyleLst>
      <a:lnStyleLst>
        <a:ln w="9525"><a:solidFill><a:schemeClr val="accent1"/></a:solidFill></a:ln>
        <a:ln w="25400"><a:solidFill><a:schemeClr val="accent2"/></a:solidFill></a:ln>
        <a:ln w="38100"><a:solidFill><a:schemeClr val="accent3"/></a:solidFill></a:ln>
      </a:lnStyleLst>
      <a:effectStyleLst>
        <a:effectStyle><a:effectLst/></a:effectStyle>
        <a:effectStyle><a:effectLst/></a:effectStyle>
        <a:effectStyle><a:effectLst/></a:effectStyle>
      </a:effectStyleLst>
      <a:bgFillStyleLst>
        <a:solidFill><a:schemeClr val="lt1"/></a:solidFill>
        <a:solidFill><a:srgbClr val="F5F1E8"/></a:solidFill>
        <a:solidFill><a:srgbClr val="E6FFFB"/></a:solidFill>
      </a:bgFillStyleLst>
    </a:fmtScheme>
  </a:themeElements>
  <a:objectDefaults/>
  <a:extraClrSchemeLst/>
</a:theme>"""


def slide_master_xml() -> str:
    return f"""{xml_header()}
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld name="CPextractor Master">
    <p:bg>
      <p:bgPr>
        <a:solidFill><a:srgbClr val="F7F7F5"/></a:solidFill>
      </p:bgPr>
    </p:bg>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst>
    <p:sldLayoutId id="1" r:id="rId1"/>
  </p:sldLayoutIdLst>
  <p:txStyles>
    <p:titleStyle><a:lvl1pPr algn="l"/></p:titleStyle>
    <p:bodyStyle><a:lvl1pPr marL="0" indent="0"/></p:bodyStyle>
    <p:otherStyle><a:defPPr/></p:otherStyle>
  </p:txStyles>
</p:sldMaster>"""


def slide_master_rels() -> str:
    return f"""{xml_header()}
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>"""


def slide_layout_xml() -> str:
    return f"""{xml_header()}
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
 type="blank" preserve="1">
  <p:cSld name="Blank">
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>"""


def slide_layout_rels() -> str:
    return f"""{xml_header()}
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>"""


def slide_rels() -> str:
    return f"""{xml_header()}
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>"""


def view_props_xml() -> str:
    return f"""{xml_header()}
<p:viewPr xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:normalViewPr horizBarState="restored" vertBarState="restored">
    <p:restoredLeft sz="15620"/>
    <p:restoredTop sz="94660"/>
  </p:normalViewPr>
  <p:slideViewPr>
    <p:cSldViewPr snapToGrid="1" snapToObjects="1"/>
  </p:slideViewPr>
  <p:notesTextViewPr/>
  <p:gridSpacing cx="72008" cy="72008"/>
</p:viewPr>"""


def pres_props_xml() -> str:
    return f"""{xml_header()}
<p:presentationPr xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:showPr loop="0" useTimings="0"/>
</p:presentationPr>"""


def table_styles_xml() -> str:
    return (
        xml_header()
        + '\n<a:tblStyleLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        + 'def="{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}"/>'
    )


def paragraph_xml(text: str, *, level: int = 0, size: int = 2200, bold: bool = False, color: str = "1F2937") -> str:
    text = escape(text)
    bullet = '<a:buChar char="•"/>' if level == 0 else '<a:buChar char="–"/>'
    mar_l = 0 if level == 0 else 342900
    indent = 0
    return f"""
      <a:p>
        <a:pPr lvl="{level}" marL="{mar_l}" indent="{indent}">{bullet}</a:pPr>
        <a:r>
          <a:rPr lang="zh-CN" sz="{size}" b="{1 if bold else 0}">
            <a:solidFill><a:srgbClr val="{color}"/></a:solidFill>
            <a:latin typeface="Aptos"/>
            <a:ea typeface="Microsoft YaHei"/>
          </a:rPr>
          <a:t>{text}</a:t>
        </a:r>
      </a:p>"""


def textbox_shape(shape_id: int, name: str, x: int, y: int, cx: int, cy: int, paragraphs: str) -> str:
    return f"""
    <p:sp>
      <p:nvSpPr>
        <p:cNvPr id="{shape_id}" name="{escape(name)}"/>
        <p:cNvSpPr txBox="1"/>
        <p:nvPr/>
      </p:nvSpPr>
      <p:spPr>
        <a:xfrm>
          <a:off x="{x}" y="{y}"/>
          <a:ext cx="{cx}" cy="{cy}"/>
        </a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        <a:noFill/>
        <a:ln><a:noFill/></a:ln>
      </p:spPr>
      <p:txBody>
        <a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0" anchor="t"/>
        <a:lstStyle/>
        {paragraphs}
      </p:txBody>
    </p:sp>"""


def rect_shape(shape_id: int, name: str, x: int, y: int, cx: int, cy: int, fill: str) -> str:
    return f"""
    <p:sp>
      <p:nvSpPr>
        <p:cNvPr id="{shape_id}" name="{escape(name)}"/>
        <p:cNvSpPr/>
        <p:nvPr/>
      </p:nvSpPr>
      <p:spPr>
        <a:xfrm>
          <a:off x="{x}" y="{y}"/>
          <a:ext cx="{cx}" cy="{cy}"/>
        </a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        <a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>
        <a:ln><a:noFill/></a:ln>
      </p:spPr>
      <p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody>
    </p:sp>"""


def slide_xml(slide: dict, idx: int) -> str:
    accent_colors = ["0F766E", "C2410C", "2563EB", "7C3AED", "4D7C0F", "B91C1C"]
    accent = accent_colors[(idx - 1) % len(accent_colors)]
    bg_band = rect_shape(2, "Top Band", 0, 0, SLIDE_W, 685800, accent)
    side_bar = rect_shape(3, "Side Accent", 548640, 1188720, 114300, 4114800, accent)

    title_paragraphs = f"""
      <a:p>
        <a:r>
          <a:rPr lang="zh-CN" sz="2800" b="1">
            <a:solidFill><a:srgbClr val="111827"/></a:solidFill>
            <a:latin typeface="Aptos Display"/>
            <a:ea typeface="Microsoft YaHei"/>
          </a:rPr>
          <a:t>{escape(slide['title'])}</a:t>
        </a:r>
      </a:p>
    """

    title_box = textbox_shape(
        4,
        "Title",
        822960,
        731520,
        10332720,
        731520,
        title_paragraphs,
    )

    body_parts = []
    if slide.get("subtitle"):
        body_parts.append(paragraph_xml(slide["subtitle"], size=2400, bold=True, color="0F172A"))
    for bullet in slide.get("bullets", []):
        body_parts.append(paragraph_xml(bullet, size=2200, color="1F2937"))
    body_parts.append("\n      <a:endParaRPr lang=\"zh-CN\" sz=\"2000\"/>")
    body_box = textbox_shape(
        5,
        "Body",
        1257300,
        1554480,
        10160040,
        4297680,
        "".join(body_parts),
    )

    footer_paragraphs = """
      <a:p>
        <a:r>
          <a:rPr lang="en-US" sz="1200">
            <a:solidFill><a:srgbClr val="475569"/></a:solidFill>
            <a:latin typeface="Aptos"/>
            <a:ea typeface="Microsoft YaHei"/>
          </a:rPr>
          <a:t>CPextractor project presentation</a:t>
        </a:r>
      </a:p>
    """
    footer_box = textbox_shape(
        6,
        "Footer",
        822960,
        6172200,
        3657600,
        228600,
        footer_paragraphs,
    )

    number_paragraphs = f"""
      <a:p>
        <a:pPr algn="r"/>
        <a:r>
          <a:rPr lang="en-US" sz="1400" b="1">
            <a:solidFill><a:srgbClr val="{accent}"/></a:solidFill>
            <a:latin typeface="Aptos"/>
            <a:ea typeface="Microsoft YaHei"/>
          </a:rPr>
          <a:t>{idx:02d}</a:t>
        </a:r>
      </a:p>
    """
    number_box = textbox_shape(
        7,
        "Slide Number",
        11216640,
        6172200,
        731520,
        228600,
        number_paragraphs,
    )

    return f"""{xml_header()}
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg>
      <p:bgPr>
        <a:solidFill><a:srgbClr val="F7F7F5"/></a:solidFill>
      </p:bgPr>
    </p:bg>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
      {bg_band}
      {side_bar}
      {title_box}
      {body_box}
      {footer_box}
      {number_box}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


def build_pptx(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    slide_count = len(SLIDES)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types(slide_count))
        zf.writestr("_rels/.rels", root_rels())
        zf.writestr("docProps/app.xml", app_xml(slide_count))
        zf.writestr("docProps/core.xml", core_xml())
        zf.writestr("ppt/presentation.xml", presentation_xml(slide_count))
        zf.writestr("ppt/_rels/presentation.xml.rels", presentation_rels(slide_count))
        zf.writestr("ppt/theme/theme1.xml", theme_xml())
        zf.writestr("ppt/slideMasters/slideMaster1.xml", slide_master_xml())
        zf.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", slide_master_rels())
        zf.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout_xml())
        zf.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", slide_layout_rels())
        zf.writestr("ppt/viewProps.xml", view_props_xml())
        zf.writestr("ppt/presProps.xml", pres_props_xml())
        zf.writestr("ppt/tableStyles.xml", table_styles_xml())
        for idx, slide in enumerate(SLIDES, start=1):
            zf.writestr(f"ppt/slides/slide{idx}.xml", slide_xml(slide, idx))
            zf.writestr(f"ppt/slides/_rels/slide{idx}.xml.rels", slide_rels())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a lightweight CPextractor project overview PPTX.")
    parser.add_argument(
        "--output",
        default="docs/presentation/CPextractor_project_overview_zh.pptx",
        help="Output .pptx file path",
    )
    args = parser.parse_args()
    out_path = Path(args.output)
    build_pptx(out_path)
    print(f"Generated PPTX -> {out_path}")


if __name__ == "__main__":
    main()
