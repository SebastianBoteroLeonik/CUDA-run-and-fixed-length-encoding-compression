all: usr/local/lib/libgtest.a

googletest-1.17.0.tar.gz:
	wget https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz

usr/local/lib/libgtest.a: googletest-1.17.0.tar.gz
	tar -xvf googletest-1.17.0.tar.gz && \
	cd googletest-1.17.0 && \
	mkdir build && \
	cd build && \
	cmake .. && \
	$(MAKE) && \
	$(MAKE) DESTDIR=../.. install

clean:
	-rm -rf googletest-1.17.0
	-rm -rf usr
