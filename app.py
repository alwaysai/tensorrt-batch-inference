import time
import edgeiq
import numpy as np


def main():
    obj_detect = edgeiq.ObjectDetection("sheshalwaysai/yolo_v3_xavier_nx")
    obj_detect.load(engine=edgeiq.Engine.TENSOR_RT)

    print("Loaded model:\n{}\n".format(obj_detect.model_id))
    print("Engine: {}".format(obj_detect.engine))
    print("Accelerator: {}\n".format(obj_detect.accelerator))
    print("Labels:\n{}\n".format(obj_detect.labels))

    fps = edgeiq.FPS()

    try:
        with edgeiq.FileVideoStream('videos/sample.mkv') as video_stream0, \
                edgeiq.Streamer(port=5005) as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            # loop detection
            while True:
                frame0 = video_stream0.read()
                frames = [frame0, frame0, frame0, frame0]

                results = obj_detect.detect_objects_batch(frames, confidence_level=.1)

                # Generate text to display on streamer
                text = ["Model: {}".format(obj_detect.model_id)]
                text.append(
                        "Inference time: {:1.3f} s".format(results[0].duration))
                text.append("Objects:")

                # Loop for markup of images with corresponding detections and text generation
                for index in range(len(frames)):
                        text.append("Results-{}".format(index))
                        frames[index] = edgeiq.markup_image(
                                frames[index], results[index].predictions, colors=obj_detect.colors)

                        for prediction in results[index].predictions:
                                text.append("{}: {:2.2f}%".format(
                                        prediction.label, prediction.confidence * 100))

                streamer.send_data(np.vstack(frames), text)

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
