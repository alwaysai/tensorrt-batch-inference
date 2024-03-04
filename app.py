import time
from typing import List
import edgeiq
import numpy as np


def stack_frames(marked_frames: List[np.ndarray]) -> np.ndarray:
    """
    Stack frames in the following arrangement:

    |  cam1  |  cam2  |
    |  cam3  |  cam4  |
    """
    stacked_frames: List[np.ndarray] = []
    for i in range(0, len(marked_frames), 2):
        if i + 1 < len(marked_frames):
            stacked_frames.append(np.hstack(marked_frames[i:i + 2]))
        else:
            # Handle odd number of streams
            last_frame = marked_frames[i]
            black_frame = np.zeros_like(last_frame)
            stacked_frames.append(np.hstack((last_frame, black_frame)))

    return np.vstack(stacked_frames)


def main():
    obj_detect = edgeiq.ObjectDetection("alwaysai/yolo_v3_xavier_nx_batch4")
    obj_detect.load(engine=edgeiq.Engine.TENSOR_RT)

    print("Loaded model:\n{}\n".format(obj_detect.model_id))
    print("Engine: {}".format(obj_detect.engine))
    print("Accelerator: {}\n".format(obj_detect.accelerator))
    print("Labels:\n{}\n".format(obj_detect.labels))

    fps = edgeiq.FPS()

    try:
        with edgeiq.FileVideoStream('videos/sample1.mp4') as video_stream0, \
                edgeiq.FileVideoStream('videos/sample2.mp4') as video_stream1, \
                edgeiq.Streamer(max_image_width=1080, max_image_height=760) as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            # loop detection
            while True:
                frame0 = video_stream0.read()
                frame1 = video_stream1.read()
                frames = [frame0, frame1, frame0, frame1]

                results = obj_detect.detect_objects_batch(frames,
                                                          confidence_level=.1)

                # Generate text to display on streamer
                text = ["Model: {}".format(obj_detect.model_id)]
                text.append(
                    "Inference time: {:1.3f} s".format(results[0].duration))
                text.append("Objects:")

                # Loop for markup of images with corresponding detections
                # and text generation
                for index in range(len(frames)):
                    text.append("Results-{}".format(index))
                    frames[index] = edgeiq.markup_image(
                        frames[index], results[index].predictions,
                        colors=obj_detect.colors)

                    for prediction in results[index].predictions:
                        text.append("{}: {:2.2f}%".format(
                                prediction.label, prediction.confidence * 100))

                streamer.send_data(stack_frames(frames), text)

                fps.update()

                if streamer.check_exit():
                    break

    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    main()
