import os
import argparse
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert image sequence to GIF")
    parser.add_argument("--imgs_dir", type=str, required=True)
    parser.add_argument("--fps", type=int, default=15, help="Target GIF fps (recommended: 10-25, max reliable: 50)")
    parser.add_argument("--source_fps", type=int, default=60, help="FPS of the original image sequence")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor (e.g. 0.5 for half size)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output dir for video.gif (default: same as imgs_dir)")
    parser.add_argument("--delete_imgs", action="store_true", default=False)

    args = parser.parse_args()

    imgs_dir = args.imgs_dir
    output_dir = args.output_dir if args.output_dir else imgs_dir
    fnames = sorted([f for f in os.listdir(imgs_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    if len(fnames) == 0:
        print("No images found in {}".format(imgs_dir))
        exit(1)

    # GIF 帧间隔以厘秒为单位，低于 20ms 的延迟在很多查看器中表现异常
    # 通过跳帧将 source_fps 降到 target fps
    skip = max(1, args.source_fps // args.fps)
    actual_fps = args.source_fps / skip
    duration_ms = int(1000 / actual_fps)
    # 确保至少 20ms，避免查看器异常
    duration_ms = max(duration_ms, 20)

    selected_fnames = fnames[::skip]
    print("Source: {} frames @ {}fps -> GIF: {} frames @ {:.1f}fps ({}ms/frame)".format(
        len(fnames), args.source_fps, len(selected_fnames), 1000 / duration_ms, duration_ms))

    pil_imgs = [Image.open(os.path.join(imgs_dir, fn)) for fn in selected_fnames]

    if args.scale != 1.0:
        pil_imgs = [
            img.resize((int(img.width * args.scale), int(img.height * args.scale)), Image.LANCZOS)
            for img in pil_imgs
        ]

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "video.gif")
    pil_imgs[0].save(
        out_path,
        save_all=True,
        append_images=pil_imgs[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )

    print("GIF saved at {}".format(out_path))

    if args.delete_imgs:
        for fn in fnames:
            os.remove(os.path.join(imgs_dir, fn))
