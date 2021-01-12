
# target folder need to be removed before each run
.phoney: all clean

all:
	python3 terub_stn.py
	rm -rf report/ target/

	python3 terub_gpe.py 
	rm -rf target/

	python3 terub_gpe_multisyn_test.py
	rm -rf target/
	
	python3 terub_stn_multisyn_test.py
	rm -rf report/ target/


clean: 
	rm -rf report/ resources/ target/