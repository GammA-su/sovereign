.PHONY: ci golden-v4 golden-v5

ci:
	./scripts/ci_all.sh

golden-v4:
	./scripts/ci_golden_suite_v4.sh

golden-v5:
	./scripts/ci_golden_suite_v5.sh
