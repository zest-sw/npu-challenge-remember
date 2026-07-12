def captioner_selector(captioner_type: str, **kwargs):
    """Factory to select a captioner (VLM backend) by name.

    - 'tt' / 'tenstorrent' / 'qwen'  -> Qwen2.5-VL on Tenstorrent NPU (HTTP server)
    - 'vila'                         -> VILA (CUDA, requires an argparse-style `args`)
    """
    name = captioner_type.lower()

    if name in ("tt", "tenstorrent", "qwen", "qwen25_vl"):
        from remembr.captioners.tenstorrent_captioner import TenstorrentCaptioner
        return TenstorrentCaptioner(**kwargs)

    if name == "vila":
        from remembr.captioners.vila_captioner import VILACaptioner
        return VILACaptioner(**kwargs)

    raise ValueError(f"Unknown captioner type: {captioner_type}")
