
.PHONY: install run run-data clean

install:
	python -m pip install -r requirements.txt

run:
	python analyze.py

run-data:
	python analyze.py --data "$(DATA)"

clean:
	rm -f assets/*.png assets/*.json
