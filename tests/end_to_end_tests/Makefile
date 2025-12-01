FILES=short_file mid_file long_file
RLE=$(patsubst %_file,%-rle,$(FILES))
FLE=$(patsubst %_file,%-fle,$(FILES))
FILE_GENERATOR=./file_generator.py
check: $(RLE) $(FLE)
	echo All end-to-end have tests passed

clean:
	-rm $(FILES:%=*%*)

short_file:
	$(FILE_GENERATOR) 50 30 50 $@
mid_file:
	$(FILE_GENERATOR) 100 100 50 $@
long_file:
	$(FILE_GENERATOR) 10000 1000 60 $@

%-rle: %_file
	./rle_test.bash $<

%-fle: %_file
	./fle_test.bash $<
