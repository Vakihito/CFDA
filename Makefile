.PHONY: run_image
run_script:
	poetry run python image_classification/main.py

.PHONY: run_qa
run_script:
	poetry run python question_answering/main.py

.PHONY: run_text
run_script:
	poetry run python text_classification/main.py