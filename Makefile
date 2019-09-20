test:
	py.test tests

.PHONY: docs
docs:
	$(MAKE) -C docs html

clean:
	python setup.py clean
	rm -rf build demosaicnet.egg-info dist .pytest_cache

distribution:
	python setup.py sdist bdist_wheel
	twine check dist/*

test_upload:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

upload_distribution:
	twine upload dist/*
