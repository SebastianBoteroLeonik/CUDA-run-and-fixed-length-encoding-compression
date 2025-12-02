# CUDA - projekt 1

## kompresja RLE i FLE

### Stuktura projektu

Projekt jest uporządkowany w katalogach. Główny kod programu znajduje się w `src/`, a pliki nagłówkowe
w `include`.

```
.
├── Makefile
├── README.md
├── include
│   ├── cli.h
│   ├── cuda_utils.cuh
│   ├── define.h
│   ├── file_io.h
│   ├── fle.h
│   └── rle.h
├── src
│   ├── cli.c
│   ├── cuda_utils.cu
│   ├── file_io.c
│   ├── fle_common.cu
│   ├── fle_compress.cu
│   ├── fle_decompress.cu
│   ├── main.c
│   ├── rle_common.cu
│   ├── rle_compress.cu
│   └── rle_decompress.cu
├── lib
│   └── ...
└── tests
    └── ...
```

Poza głównym kodem pisałem testy, choć z uwagi na fakt, że nie były one wymagane,
tak nie dbałem o jakość kodu w testach. Głównie obchodziło mnie, żeby spełniały swoją powinność.

### Kompilacja

W projekcie używany jest `make`

Oto główne targety:

- `make` - skompilowanie programu i umieszczenie go w `bin/`
- `make check` - skompilowanie i odpalenie testów - jednostkowych i end-to-end
- `make clean` - usunięcie plików powstałych w budowie

Do testów jednostkowych używany jest [gtest](https://google.github.io/googletest/),
który wymaga `cmake` do kompilacji. W testach end-to-end wykorzystałem `python` z `numpy`
do wygenerowania plików testowych

#### Makra

Makra do kompilacji można przekazywać do `make` poprzez zmienną `DEFINE`.

Aby skompilować program w wersji nie wykorzystującej kartę graficzną należy zdefiniować makro:
`DEFINE=-DCPU`

Pod koniec udało mi się zaimplementować jeden element programu w wersji wykorzystującej strumienie CUDA
(Nie był to specjalnie udany eksperyment, ale działa poprawnie, choć nie za szybko).
Aby otrzymać program, który kompiluje rle w wersji dwustrumieniowej należy przekazać
`DEFINE=-DSTREAMS`

Choć pewnie profiler jest lepszą opcją do sprawdzania czasu wykonania funkcji na karcie, napisałem bardzo
proste sprawdzanie tych czasów za pomocą cudaEvent_t.
Aby te informacje się wyświetlały należy zdefiniować
`DEFINE=-DPERFORMANCE_TEST`

### Ogólny zarys działania

#### RLE

W RLE mamy do czynienia z operacją Pack, zrealizowaną za pomocą Prescan. Dość klasyczny przypadek po
wykładzie, ale wpadłem na algorytm sam przed wykładem, z czego jestem dumny.

Jedynym problemem względem klasycznego Pack jest drugi etap, w którym należy rozbić komórki,
które przekroczyły 256 powtórzeń (używam jeden bajt do zapisu liczby powtórzeń).
Aby to uzyskać, wykorzystuje nowego Scana na tablicy, która nam mówi ilukrotnie przekroczyliśmy to ograniczenie.

W ten sposób otrzymuję przesunięcia, jakie muszę wykonać na tablicy z pierwszego etapu.

W dekompresji jest prosto, jako że wystarczy zrobić skan tablicy powtórzeń.

#### FLE

W FLE dzielę tablicę na fragmenty po 1024 bajty.
dla każdego z tych fragmentów otrzymanie liczby bitów potrzebnych jest trywialne ze względu na operację
`__syncthreads_or`.
Później należy jedynie poprzesuwać bity.

Dekompresja jedynie przesuwa bity.

Poniewać każdy fragment zajmuje różną wielkość, w strukturze jest dodatkowa tablica z wielkościami
tychże fragmentów.

### Komentarze

Głównie starałem się okomentować kod w plikach nagłówkowych.

Większość komentarzy jest po angielsku, są to komentarze dokumentujące kod.
Komentarze po polsku są niejakim dodatkiem, który jest raczej ukierunkowany dla sprawdzającego,
żeby wyjaśnić osobliwe fragmenty kodu.

Starałem się też prowadzić _w miarę_ sensowną historię commitów w gicie, więc _powinno_ to być
sensownym źródłem informacji o projekcie. Ręki sobie nie dam uciąć.

[Repozytorium](https://github.com/SebastianBoteroLeonik/CUDA-run-and-fixed-length-encoding-compression)
