CURRENT_DATA := data/rap
PROJECT := 'rap'


.PHONY: oyomi
oyomi:
	cd $(CURRENT_DATA) && cat raw.txt | mecab -Oyomi > raw_yomi.txt


.PHONY: preprocess
preprocess:
	@# Create inputs and outputs then vectorize them.
	python3 preprocess.py


.PHONY: train
train:
	python3 train.py


.PHONY: aws-ls
aws-ls:
	aws s3 ls s3://nlp-concierge/data/ --recursive --human-readable --summalize


.PHONY: aws-up
aws-up:
	aws s3 sync ./data s3://nlp-concierge/data
	aws s3 sync ./models s3://nlp-concierge/models


.PHONY: aws-down
aws-down:
	aws s3 sync s3://nlp-concierge/data ./data
	aws s3 sync s3://nlp-concierge/models ./models


.PHONY: test
test:
	@# test
	@echo 'hello'
