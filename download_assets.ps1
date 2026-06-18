# 在有网的机器上运行此脚本，下载所有前端依赖到 web/ 目录
# 用法：在项目根目录执行  .\download_assets.ps1

$webDir = Join-Path $PSScriptRoot "web"
$fontsDir = Join-Path $webDir "fonts"
New-Item -ItemType Directory -Force -Path $fontsDir | Out-Null

Write-Host "正在下载前端资源到 $webDir ..."

$downloads = @(
    @{ Url = "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js";   Dest = "highlight.min.js" },
    @{ Url = "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css"; Dest = "highlight-theme.css" },
    @{ Url = "https://cdn.jsdelivr.net/npm/marked@9.1.6/marked.min.js";                       Dest = "marked.min.js" },
    @{ Url = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js";                  Dest = "katex.min.js" },
    @{ Url = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css";                 Dest = "katex.min.css" },
    @{ Url = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js";    Dest = "auto-render.min.js" }
)

foreach ($d in $downloads) {
    $dest = Join-Path $webDir $d.Dest
    Write-Host "  $($d.Dest) ..."
    Invoke-WebRequest -Uri $d.Url -OutFile $dest -UseBasicParsing
}

# KaTeX 字体（公式渲染必需）
Write-Host "  KaTeX 字体..."
$fonts = @(
    "KaTeX_AMS-Regular.woff2","KaTeX_Caligraphic-Bold.woff2","KaTeX_Caligraphic-Regular.woff2",
    "KaTeX_Fraktur-Bold.woff2","KaTeX_Fraktur-Regular.woff2",
    "KaTeX_Main-Bold.woff2","KaTeX_Main-BoldItalic.woff2","KaTeX_Main-Italic.woff2","KaTeX_Main-Regular.woff2",
    "KaTeX_Math-BoldItalic.woff2","KaTeX_Math-Italic.woff2",
    "KaTeX_SansSerif-Bold.woff2","KaTeX_SansSerif-Italic.woff2","KaTeX_SansSerif-Regular.woff2",
    "KaTeX_Script-Regular.woff2",
    "KaTeX_Size1-Regular.woff2","KaTeX_Size2-Regular.woff2","KaTeX_Size3-Regular.woff2","KaTeX_Size4-Regular.woff2",
    "KaTeX_Typewriter-Regular.woff2"
)
$base = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/fonts/"
foreach ($f in $fonts) {
    Invoke-WebRequest -Uri "$base$f" -OutFile (Join-Path $fontsDir $f) -UseBasicParsing
}

Write-Host ""
Write-Host "完成！文件清单："
Get-ChildItem $webDir -Recurse -File | Select-Object -ExpandProperty FullName | ForEach-Object { "  $_" }
