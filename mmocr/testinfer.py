from mmocr.apis import MMOCRInferencer

ocr = MMOCRInferencer(det='DBNet', rec='CRNN', device='cpu')
ocr('demo/demo_text_ocr.jpg', show=True, print_result=True)