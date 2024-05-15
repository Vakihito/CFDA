.PHONY: run_image
run_image:
	poetry run python image_classification/main.py

.PHONY: run_qa
run_qa:
	poetry run python question_answering/main.py

.PHONY: run_text
run_text:
	poetry run python text_classification/main.py