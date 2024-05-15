.PHONY: run_image
run_image:
	python image_classification/main.py

.PHONY: run_qa
run_qa:
	python question_answering/main.py

.PHONY: run_text
run_text:
	python text_classification/main.py