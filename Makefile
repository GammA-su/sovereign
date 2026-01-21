.PHONY: ci golden-v4 golden-v5 golden-v8 golden-v8-replay

ci:
	./scripts/ci_all.sh

golden-v4:
	./scripts/ci_golden_suite_v4.sh

golden-v5:
	./scripts/ci_golden_suite_v5.sh

golden-v8:
	./scripts/ci_golden_suite_v8.sh

golden-v8-replay:
	./scripts/ci_golden_suite_v8_replay.sh
