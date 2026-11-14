PY ?= python
export PYTHONPATH := src

.PHONY: data competitor conversion renewal-fit elasticity optimisation retention governance sql notebooks reports app test verify all

data:
	$(PY) -m retailpricing.data_gen.run
competitor:
	$(PY) -m retailpricing.competitor.run
conversion:
	$(PY) -m retailpricing.conversion.run
renewal-fit:
	$(PY) -m retailpricing.retention.fit
elasticity:
	$(PY) -m retailpricing.elasticity.run
optimisation:
	$(PY) -m retailpricing.optimisation.run
retention:
	$(PY) -m retailpricing.retention.run
governance:
	$(PY) -m retailpricing.governance.run
sql:
	$(PY) -m retailpricing.analytics_sql.duck
notebooks:
	$(PY) scripts/build_notebooks.py
reports:
	$(PY) -m retailpricing.reporting.build
app:
	streamlit run app/streamlit_app.py
test:
	$(PY) -m pytest -q --junitxml=data/processed/pytest_junit.xml
verify:
	$(PY) scripts/verify.py

# Dependency order: data, competitor, conversion, renewal-fit, elasticity, optimisation, retention, governance, sql
all: data competitor conversion renewal-fit elasticity optimisation retention governance sql test reports notebooks verify
