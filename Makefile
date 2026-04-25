.PHONY: build-dashboard publish lint typecheck test

build-dashboard:
	cd dashboard && npm install && npm run build
	cp -r dashboard/dist/* agenteval/dashboard_dist/
	@echo "Dashboard built and copied to agenteval/dashboard_dist/"

publish:
	cd dashboard && npm install && npm run build
	cp -r dashboard/dist/* agenteval/dashboard_dist/
	python -m build
	twine check dist/*
	twine upload dist/*

lint:
	ruff check .

typecheck:
	mypy agenteval/

test:
	pytest --cov=agenteval --cov-report=term-missing
