import os
import argparse
import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--imgs_dir", type=str, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default=None, help="Output dir for video.mp4 (default: same as imgs_dir)")
    parser.add_argument("--delete_imgs", action="store_true", default=False)

    args = parser.parse_args()

    imgs_dir = args.imgs_dir
    output_dir = args.output_dir if args.output_dir else imgs_dir
    fps = args.fps
    delete_imgs = args.delete_imgs

    fnames = os.listdir(imgs_dir)
    fnames = sorted(fnames)

    imgs = [cv2.imread(os.path.join(imgs_dir, fn), cv2.IMREAD_COLOR) for fn in fnames]

    os.makedirs(output_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    size = imgs[0].shape[:2]
    videoWriter = cv2.VideoWriter(os.path.join(output_dir, "video.mp4"), fourcc, fps, (size[1], size[0]))

    for i in range(len(imgs)):
        # cv2.imshow("test", imgs[i])
        # cv2.waitKey(0)
        videoWriter.write(imgs[i])
    
    videoWriter.release()

    print("video saved at {}".format(os.path.join(output_dir, "video.mp4")))

    if delete_imgs:
        for i in range(len(imgs)):
            os.remove(os.path.join(imgs_dir, fnames[i]))
