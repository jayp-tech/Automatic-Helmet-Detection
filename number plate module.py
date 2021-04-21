def numberplate( int [] ):
    try:
        with detection_graph.as_default():
            with tf.Session() as sess:
                    # Get handles to input and output tensors
                    ops = tf.get_default_graph().get_operations()
                    all_tensor_names = {output.name for op in ops for output in op.outputs}
                    tensor_dict = {}
                    for key in [
                        'num_detections', 'detection_boxes', 'detection_scores',
                      'detection_classes', 'detection_masks'
                    ]:
                        tensor_name = key + ':0'
                        if tensor_name in all_tensor_names:
                            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                          tensor_name)

                    
                    ret, image_np = cap.read()
                    ret = cap.set(3,1080)
                    ret = cap.set(4,720)
                    frame = cv2.flip(image_np, 1)
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    # Actual detection.
                    imagenp2 =cv2.resize(image_np, (800, 600))
                    output_dict = run_inference_for_single_image(imagenp2, detection_graph)
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        output_dict['detection_boxes'],
                        output_dict['detection_classes'],
                        output_dict['detection_scores'],
                        category_index,
                        instance_masks=output_dict.get('detection_masks'),
                        use_normalized_coordinates=True,
                        line_thickness=8)
                    print("44")
                    (height , width) = image_np.shape[:2]
                    cv2.imshow('object_detection', imagenp2)
                    if output_dict['detection_scores'][0] > 0.80:
                        crop_img = image_np[int((output_dict['detection_boxes'][0][0]) * 480): int(
                                    (output_dict['detection_boxes'][0][2]) * 480),
                                    int((output_dict['detection_boxes'][0][1]) * 640):int(
                                    (output_dict['detection_boxes'][0][3]) * 640)]
                        print("got here")
                        cv2.imshow("cropped image",crop_img)
                        cv2.imwrite("original.jpg", cv2.resize(image_np, (800, 600)))
                        cv2.imwrite("cropped.jpg", crop_img)
                        cap.release()
                        cv2.destroyAllWindows()
    except Exception as e:
        print(e)
        print("excp")
        cap.release()
        cv2.destroyAllWindows()


    crop_img = image_np[int((output_dict['detection_boxes'][0][0]) * 480) +10 : int(
                                    (output_dict['detection_boxes'][0][2]) * 480) -10,
                                    int((output_dict['detection_boxes'][0][1]) * 640) +20 :int(
                                    (output_dict['detection_boxes'][0][3]) * 640) -20]
                        #print("got here")
    cv2.imwrite("crop.jpg",crop_img)


    import pytesseract
    from PIL import Image

    img = Image.open('crop.jpg')
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    numplatestring = pytesseract.image_to_string(img)
    return numplatestring