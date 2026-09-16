"""Download the 2020 census table 1-4 (population by region, sex and nationality) per province.

Every source here is a provincial statistics bureau's own 2020 census yearbook, or the
Wayback Machine's copy of one where the live host refuses scripts from outside China. The
national table is the same table at province level and is what parse.py checks every
province against.

Most of these yearbooks list only a JPG for each table in their index (`left.htm`), but an
Excel copy sits beside the image under the same name. Nothing links to it. See NOTES.md.

Usage:
    python fetch.py            # everything missing
    python fetch.py --force    # re-download everything
    python fetch.py henan      # one or more by key
"""
import argparse
import os
import shutil
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CURL = shutil.which("curl") or r"C:\Windows\System32\curl.exe"

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw", "2020")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

WB = "https://web.archive.org/web/{ts}if_/{url}"

# key: (GB province code, file name, url, note)
SOURCES = {
    "national": ("00", "national_A0104.xls",
                 "https://www.stats.gov.cn/sj/pcsj/rkpc/7rp/zk/html/A0104.xls",
                 "NBS China Population Census Yearbook 2020, table 1-4, province level"),
    "beijing": ("11", "beijing_A0106.xls",
                "https://nj.tjj.beijing.gov.cn/tjnj/rkpc-2020/e/zk/html/A0106.xls",
                "Beijing numbers this table 1-6"),
    "neimenggu": ("15", "neimenggu_A0104.xls",
                  WB.format(ts="20240509104832", url="https://tj.nmg.gov.cn/files_pub/content/"
                            "PAGEPACK/zk2020/html/A0104.xls"),
                  "live host returns a cloud-protection 403 from outside China"),
    "jilin": ("22", "jilin_A0104.xls",
              "https://tjj.jl.gov.cn/tjsj/qwfb/jlsdqcqgrkpcnj/zk/html/A0104.xls", ""),
    "heilongjiang": ("23", "heilongjiang_A0104.xls",
                     "https://tjj.hlj.gov.cn/tjjnianjian/2020rkpc/zk/html/A0104.xls", ""),
    "shanghai": ("31", "shanghai_ANJ-1-04.xls",
                 "https://tjj.sh.gov.cn/tjnj/2020rktjnj/ANJ-1-04.xls", ""),
    "jiangsu": ("32", "jiangsu_A0104.xls",
                "https://tj.jiangsu.gov.cn/2020pcnj/zk/html/A0104.xls", ""),
    "zhejiang": ("33", "zhejiang_A0104.xls",
                 WB.format(ts="20260619055205",
                           url="https://zjjcmspublic.oss-cn-hangzhou-zwynet-d01-a.internet.cloud."
                               "zj.gov.cn/jcms_files/jcms1/web3077/site/flash/tjj/Reports1/"
                               "%E6%B5%99%E6%B1%9F%E4%BA%BA%E6%99%AE2020%E5%85%89%E7%9B%98/"
                               "zk/html/A0104.xls"),
                 "live file host returns 403 to scripts"),
    "fujian": ("35", "fujian_a0104.xls",
               "https://tjj.fujian.gov.cn/tongjinianjian/rk2020/html/a0104.xls",
               "no zk/ in the path, lower-case a0104"),
    "shandong": ("37", "shandong_A0104.xls",
                 "http://tjj.shandong.gov.cn/pcsj/2020/zk/html/A0104.xls",
                 "http only; the https certificate is wrong"),
    "henan": ("41", "henan_A0104.xls",
              "https://oss.henan.gov.cn/sbgt-wztipt/attachment/hntjj/hntj/lib/tjnj/"
              "2020hnsrkpcnj/zk/html/A0104.xls", ""),
    "hubei": ("42", "hubei_2020.zip",
              "https://tjj.hubei.gov.cn/tjsj/sjkscx/tjzl/202303/P020230824390723778530.zip",
              "whole yearbook as xlsx; table 1-4 is one member"),
    "guangxi": ("45", "guangxi_A1-04.xls",
                "http://tjj.gxzf.gov.cn//tjsj/tjsj_ztsj/material/"
                "2020%E5%B9%B4%E5%B9%BF%E8%A5%BF%E4%BA%BA%E5%8F%A3%E6%99%AE%E6%9F%A5"
                "%E5%B9%B4%E9%89%B4/zk/html/A1-04.xls",
                "named A1-04; A1-4a is a household-registration version, not this"),
    "hainan": ("46", "hainan_A0104.xls",
               "https://stats.hainan.gov.cn/tjj/2023nj/zk/html/A0104.xls", ""),
    "chongqing": ("50", "chongqing_2020.pdf",
                  "https://tjj.cq.gov.cn/cslm/tjsjzl/202407/P020240725646306699998.pdf",
                  "whole yearbook as one 337 MB PDF"),
    "yunnan": ("53", "yunnan_2020.rar",
               "https://stats.yn.gov.cn/zwgk/zfxxgk/fdzdgknr/tjsj/tjnj/202607/"
               "P020260716686337458968.rar",
               "whole yearbook as xlsx in a rar"),
    "qinghai": ("63", "qinghai_A0104.xls",
                WB.format(ts="20250216153853",
                          url="http://tjj.qinghai.gov.cn/nj/rkpc/zk/html/A0104.xls"),
                "live host sends a JavaScript bot challenge"),
    "ningxia": ("64", "ningxia_rkpc2021.pdf",
                WB.format(ts="20260527125321",
                          url="https://nxdata.com.cn/files_nx_pub/pdf/pdf/rkpc2021/rkpc2021.pdf"),
                "whole yearbook as a text PDF; table 1-4 is PDF pages 30-49"),
}

# The same table where it stops at prefectures. Not drawn as counties: these are the 2020
# totals that fallback.py scales an older county pattern to.
PREFECTURE_SOURCES = {
    "hebei": ("13", "hebei_A0104.xls",
              "https://tjj.hebei.gov.cn/extra/col20/rkpc2020/zk/html/A0104.xls", ""),
    "liaoning": ("21", "liaoning_A0104.xls",
                 WB.format(ts="20250226002531",
                           url="https://tjj.ln.gov.cn/tjj/tjxx/pcsj/people/pczl/zk/html/A0104.xls"),
                 "live host blocks foreign IPs"),
    "hunan": ("43", "hunan_A1-04.xls",
              "http://222.240.193.190/2020rkpcnj/html/A1-04.xls", "bare IP; may not last"),
    "sichuan": ("51", "sichuan_A0104.xls",
                "https://tjj.sc.gov.cn/scstjj/rkpcnew/2020/zk/html/A0104.xls", ""),
}
SOURCES.update(PREFECTURE_SOURCES)

MAGIC = {".xls": b"\xd0\xcf\x11\xe0", ".zip": b"PK\x03\x04", ".rar": b"Rar!",
         ".pdf": b"%PDF"}


def check(path, ext):
    """A bot wall returns HTML with HTTP 200, so trust the bytes, not the status."""
    with open(path, "rb") as fh:
        head = fh.read(8)
        if ext == ".pdf":
            fh.seek(max(0, os.path.getsize(path) - 1024))
            if b"%%EOF" not in fh.read():
                return "PDF has no %%EOF trailer (truncated)"
    if not head.startswith(MAGIC[ext]):
        return f"not a {ext} file; starts {head!r}"
    return None


def fetch(key, force=False):
    code, fn, url, note = SOURCES[key]
    dest = os.path.join(RAW, fn)
    ext = os.path.splitext(fn)[1].lower()
    if os.path.exists(dest) and not force and check(dest, ext) is None:
        print(f"  {key:13s} have {fn} ({os.path.getsize(dest):,} bytes)")
        return True
    tmp = dest + ".part"
    if force and os.path.exists(tmp):
        os.remove(tmp)
    # curl rather than urllib: Python 3.9's bundled certificates reject several of these
    # hosts ("self signed certificate in certificate chain") where curl, using the
    # Windows certificate store, connects fine. The bytes are checked below either way.
    # -C - resumes a .part left by an earlier run: Chongqing's host drops the connection
    # partway through its 337 MB PDF, usually more than once.
    cmd = [CURL, "-sS", "-L", "--fail", "-C", "-", "--retry", "3", "--retry-delay", "5",
           "--max-time", "3600", "-A", UA["User-Agent"], "-o", tmp, url]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(tmp):
        print(f"  {key:13s} !! curl exit {r.returncode}: {r.stderr.strip()[:200]}")
        if r.returncode == 18 and os.path.exists(tmp):
            print(f"  {key:13s}    kept {os.path.getsize(tmp):,} bytes; run again to resume")
        elif os.path.exists(tmp):
            os.remove(tmp)
        return False
    bad = check(tmp, ext)
    if bad:
        print(f"  {key:13s} !! {bad}")
        os.remove(tmp)
        return False
    os.replace(tmp, dest)
    print(f"  {key:13s} got  {fn} ({os.path.getsize(dest):,} bytes)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)
    keys = args.keys or list(SOURCES)
    failed = [k for k in keys if not fetch(k, args.force)]
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
