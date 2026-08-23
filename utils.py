def setup_matplotlib():
    """
    1. 先尝试下载并注册 Noto Sans SC
    2. 成功则用下载的字体
    3. 失败则回退 fc-list 探测
    4. 再失败用常见预装字体名
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import subprocess
    import tempfile
    import urllib.request

    print("[FONT] === setup_matplotlib START ===")
    print(f"[FONT] Matplotlib version: {matplotlib.__version__}")
    print(f"[FONT] Font cache dir: {fm.get_cachedir()}")

    # 1. 优先：下载并注册（最可靠，无需预装字体文件）
    def _download_and_register_font():
        """
        运行时下载 Noto Sans SC Regular (TTF) → 写入临时文件 → 用 font_manager.addfont 注册。
        返回注册成功的字体家族名，失败返回 None。
        """
        # Google Fonts CDN：Noto Sans SC Regular (TTF 子集，约 1.2MB)
        font_url_ttf = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/TTF/SimplifiedChinese/NotoSansSC-Regular.ttf"
        font_url_otf = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansSC-Regular.otf"

        for url in (font_url_ttf, font_url_otf):
            try:
                print(f"[FONT] Downloading font from {url} ...")
                with urllib.request.urlopen(url, timeout=15) as resp:
                    font_data = resp.read()
                print(f"[FONT] Downloaded {len(font_data)} bytes")

                # 写入临时文件
                with tempfile.NamedTemporaryFile(suffix=".ttf", delete=False) as tmp:
                    tmp.write(font_data)
                    tmp_path = tmp.name
                print(f"[FONT] Saved to {tmp_path}, size={os.path.getsize(tmp_path)}")

                # 注册到 matplotlib
                fm.fontManager.addfont(tmp_path)
                # 获取字体家族名
                prop = fm.FontProperties(fname=tmp_path)
                family = prop.get_name()
                print(f"[FONT] ✅ Registered font family: {family}")
                return family

            except Exception as e:
                print(f"[FONT] ⚠️ Failed to download/register from {url}: {e}")
                continue

        return None

    print("[FONT] === setup_matplotlib START ===")
    print(f"[FONT] Matplotlib version: {matplotlib.__version__}")
    print(f"[FONT] Font cache dir: {fm.get_cachedir()}")

    # 1. 优先：下载并注册（最可靠，无需预装字体文件）
    family = _download_and_register_font()
    if family:
        plt.rcParams["font.family"] = family
        print(f"[FONT] ✅ Using downloaded font: {family}")
        plt.rcParams["axes.unicode_minus"] = False
        print("[FONT] === setup_matplotlib END (downloaded) ===")
        return plt

    # 2. 回退：fc-list 探测系统中文字体
    try:
        out = subprocess.check_output(
            ["fc-list", ":lang=zh", "family"],
            stderr=subprocess.DEVNULL, text=True, timeout=3
        )
        for line in out.splitlines():
            fam = line.split(":")[0].strip()
            if fam:
                plt.rcParams["font.family"] = fam
                print(f"[FONT] fc-list detected: {fam}")
                plt.rcParams["axes.unicode_minus"] = False
                print("[FONT] === setup_matplotlib END (fc-list) ===")
                return plt
    except Exception as e:
        print(f"[FONT] fc-list failed: {e}")

    # 3. 兜底：常见预装字体名
    for fname in ["Noto Sans CJK SC", "Noto Sans CJK", "Source Han Sans SC", "WenQuanYi Zen Hei"]:
        try:
            plt.rcParams["font.family"] = fname
            print(f"[FONT] Fallback to preset: {fname}")
            plt.rcParams["axes.unicode_minus"] = False
            print("[FONT] === setup_matplotlib END (preset) ===")
            return plt
        except Exception as e:
            print(f"[FONT] Preset {fname} failed: {e}")

    # 4. 终极兜底：DejaVu Sans（不支持中文但不报错）
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    print("[FONT] ⚠️ Using DejaVu Sans (no Chinese support)")
    print("[FONT] === setup_matplotlib END (DejaVu) ===")
    return plt
