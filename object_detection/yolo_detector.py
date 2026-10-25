import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import onnxruntime as ort
import time
import os

# COCO classes (YOLOv8 standard)
COCO_CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
    'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
    'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
    'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
    'remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator',
    'book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]

# Color palette for different classes
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype=np.uint8)


def nms(boxes, scores, iou_threshold=0.45):
    """Non-Maximum Suppression."""
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


class YOLOv8Detector(Node):
    def __init__(self):
        super().__init__('yolov8_detector')

        # Declare parameters
        self.declare_parameter('model_path', '/home/lenovo/ros2_ws/src/object_detection/object_detection/yolov8.onnx')
        self.declare_parameter('conf_threshold', 0.35)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('input_size', 640)

        # Get parameters
        model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        self.input_size = self.get_parameter('input_size').value

        self.bridge = CvBridge()

        # Subscribe to camera topic
        self.subscription = self.create_subscription(
            Image,
            'camera',
            self.callback,
            10
        )

        # Initialize ONNX Runtime session with optimizations
        self.get_logger().info(f"Loading YOLOv8 model from: {model_path}")
        
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 4
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=['CPUExecutionProvider']
        )
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Get input shape to verify model
        input_shape = self.session.get_inputs()[0].shape
        self.get_logger().info(f"Model input shape: {input_shape}")
        self.get_logger().info(f"Model output names: {self.output_names}")
        self.get_logger().info(f"YOLOv8 model loaded successfully!")
        self.get_logger().info(f"Confidence threshold: {self.conf_threshold}")
        self.get_logger().info(f"IOU threshold: {self.iou_threshold}")

        # Performance tracking
        self.frame_count = 0
        self.total_time = 0.0

        # Dataset frame-saving setup
        self.dataset_dir = '/home/lenovo/dataset_frames'
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        # Create window with proper flags for maximize button
        self.window_name = "YOLOv8 Detection"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        
        # Optional: Set initial window size (uncomment if needed)
        # cv2.resizeWindow(self.window_name, 1280, 720)

    def preprocess(self, frame):
        """Preprocess frame with letterbox resizing."""
        h, w = frame.shape[:2]
        
        # Letterbox resize
        r = min(self.input_size / h, self.input_size / w)
        new_h, new_w = int(h * r), int(w * r)
        
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        top = (self.input_size - new_h) // 2
        left = (self.input_size - new_w) // 2
        padded[top:top+new_h, left:left+new_w] = resized
        
        # Normalize and transpose
        img = padded.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        
        return img, r, top, left

    def postprocess_v8(self, outputs, orig_h, orig_w, ratio, pad_top, pad_left):
        """
        Postprocess YOLOv8 output.
        YOLOv8 output format: [batch, 84, 8400] where 84 = 4 (box) + 80 (classes)
        """
        output = outputs[0]
        
        # YOLOv8 output is [1, 84, 8400] - need to transpose to [8400, 84]
        if len(output.shape) == 3:
            output = output[0].T  # [8400, 84]
        
        boxes = []
        scores = []
        class_ids = []

        # Process each detection
        for detection in output:
            # First 4 values are box coordinates (cx, cy, w, h)
            cx, cy, w, h = detection[:4]
            
            # Remaining values are class scores
            class_scores = detection[4:]
            
            # Get the class with highest score
            cls_id = int(np.argmax(class_scores))
            conf = float(class_scores[cls_id])
            
            # Filter by confidence
            if conf < self.conf_threshold:
                continue
            
            # Remove padding and scale back to original image
            cx = (cx - pad_left) / ratio
            cy = (cy - pad_top) / ratio
            w = w / ratio
            h = h / ratio
            
            # Convert to corner coordinates
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            
            # Clip to image boundaries
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            
            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            class_ids.append(cls_id)

        return boxes, scores, class_ids

    def draw_detections(self, frame, boxes, scores, class_ids, keep_indices):
        """Draw bounding boxes and labels."""
        for idx in keep_indices:
            x1, y1, x2, y2 = boxes[idx]
            cls = class_ids[idx]
            conf = scores[idx]

            # Get color for this class
            color = tuple(map(int, COLORS[cls]))

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"{COCO_CLASSES[cls]} {conf:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw label background
            cv2.rectangle(
                frame,
                (x1, y1 - label_h - baseline - 5),
                (x1 + label_w, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame, label, (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

        return frame

    def callback(self, msg):
        """Process incoming image messages."""
        start_time = time.time()

        # Convert ROS Image to OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        orig_h, orig_w = frame.shape[:2]

        # Preprocess
        blob, ratio, pad_top, pad_left = self.preprocess(frame)

        # Inference
        outputs = self.session.run(self.output_names, {self.input_name: blob})

        # Postprocess
        boxes, scores, class_ids = self.postprocess_v8(
            outputs, orig_h, orig_w, ratio, pad_top, pad_left
        )

        # Apply NMS
        keep_indices = nms(boxes, scores, self.iou_threshold)

        # Draw detections
        frame = self.draw_detections(frame, boxes, scores, class_ids, keep_indices)

        # Save raw (unannotated) frame every 10th frame for dataset labeling
        if self.frame_count % 10 == 0:
            raw_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            save_path = os.path.join(self.dataset_dir, f'frame_{self.frame_count:05d}.jpg')
            cv2.imwrite(save_path, raw_frame)

        # Calculate and display FPS
        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0
        
        cv2.putText(
            frame, f"YOLOv8 | FPS: {fps:.1f} | Objects: {len(keep_indices)}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        # Display with the named window
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)

        # Log performance
        self.frame_count += 1
        self.total_time += elapsed
        if self.frame_count % 100 == 0:
            avg_fps = self.frame_count / self.total_time
            self.get_logger().info(f"Average FPS: {avg_fps:.2f}, Detections: {len(keep_indices)}")

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main():
    rclpy.init()
    try:
        node = YOLOv8Detector()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
