FILES=short_file mid_file long_file
RLE=$(patsubst %_file,%-rle,$(FILES))
FLE=$(patsubst %_file,%-fle,$(FILES))
check: $(RLE) $(FLE)
	echo All end-to-end have tests passed

clean:
	-rm $(FILES:%=*%*)

short_file:
	./mock_file.py 50 30 50 $@
mid_file:
	./mock_file.py 100 100 50 $@
long_file:
	./mock_file.py 10000 1000 60 $@

%-rle: %_file
	./rle_test.bash $<

%-fle: %_file
	./fle_test.bash $<
