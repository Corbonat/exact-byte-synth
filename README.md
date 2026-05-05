# Exact Byte Synth

Система автоматически подбирает компактную арифметическую формулу для фиксированной 8-битной функции, проверяет ее на всех входах и генерирует Verilog-модуль для запуска на FPGA.

## Часть 1. Описание Проекта

Многие небольшие функции от 8-битного входа встречаются в аппаратных задачах: простая обработка сигналов, коррекция значений датчиков, битовые преобразования, кодирование или демонстрационные математические функции. Прямой способ реализовать такую функцию на FPGA - положить все 256 значений в lookup table. Это удобно, но не всегда экономно: вместо небольшой формулы мы используем память.

Идея проекта - решить обратную задачу. Мы считаем целевую функцию черным ящиком: для каждого `x` от `0` до `255` можно узнать `f(x)`, но сама формула заранее неизвестна. Программа на компьютере перебирает простые выражения над `x`, ищет выражение с тем же поведением на всех 256 входах, а затем превращает найденную формулу в Verilog.

### Как Это Работает

На вход подается функция-oracle: программа может вычислить `f(x)` для любого байта, но не знает формулы заранее. Синтезатор строит кандидатов из простых арифметических и битовых операций, сравнивает их поведение со значениями oracle на всех 256 входах и выбирает выражение, которое совпадает полностью.

После этого найденная формула проходит проверку и может быть записана как Verilog-модуль. Подробное описание поиска, ограничений и научных референсов вынесено в `ALGORITHM.md`.

```mermaid
flowchart LR
    A["Target function<br/>black-box oracle"] -->|evaluate f(x), x=0..255| B["Truth table<br/>256 bytes"]
    B --> C["C++ synthesizer<br/>bottom-up search"]
    C --> D["Exact expression<br/>compact formula"]
    D --> E["Full-domain verification<br/>0..255"]
    E --> F["Verilog module<br/>generated_function.v"]
    F --> G["FPGA top-level<br/>switches, LEDs, HEX displays"]
```



## Часть 2. Гайд По Использованию

Основной сценарий рассчитан на Windows и PowerShell. Для сборки нужен `g++` с поддержкой C++20, например MSYS2 UCRT64. Python нужен только для дополнительных benchmark-скриптов, Quartus - только для запуска FPGA-части на плате.

### Сборка И Тест

```powershell
powershell -ExecutionPolicy Bypass -File cpp_port\build.ps1
cpp_port\build\synth_tests.exe
```

После сборки все `.exe` лежат в `cpp_port\build\`.

### Своя Функция

Чтобы синтезировать свою функцию, поменяйте `current_oracle` в `cpp_port/src/current_oracle.cpp`:

```cpp
std::uint8_t current_oracle(std::uint8_t x) {
    return static_cast<std::uint8_t>(((x ^ 17) + (x >> 2)) & MASK);
}
```

Затем пересоберите проект и запустите поиск:

```powershell
powershell -ExecutionPolicy Bypass -File cpp_port\build.ps1
cpp_port\build\synth_search.exe --max-cost 6 --nonlinear-mode restricted
```

### Verilog И FPGA

Сгенерировать Verilog для текущего oracle:

```powershell
cpp_port\build\synth_codegen.exe --output generated_function.sv --module-name generated_function
```

Сгенерировать Verilog для готового benchmark-кейса и положить его в FPGA-папку:

```powershell
cpp_port\build\synth_codegen.exe --benchmark xor_shift_add --output fpga_demo\generated_function.sv
```

После запуска основной результат будет здесь:

- `fpga_demo/generated_function.sv` - активный Verilog-модуль для подключения к FPGA top-level.

Чтобы собрать это в Quartus, создайте проект в папке `fpga_demo/`, выберите top-level entity `top` и добавьте файлы `top.sv`, `hex_to_7seg.sv`, `generated_function.sv`. Затем импортируйте или используйте настройки из `fpga_demo/exact-byte-synth.qsf`.

Файл `exact-byte-synth.qsf` написан под конкретную плату: в нем выбран FPGA `Cyclone IV E EP4CE6E22C8` и назначены физические пины. Он также ожидает конкретные имена портов top-level модуля: `clk`, `sw`, `led`, `segments`, `digit_en`. Для другой платы этот `.qsf` нельзя использовать как универсальный: нужно заменить `DEVICE`, pin assignments и привести `top.sv` и `.qsf` к одним и тем же сигналам.

### Benchmark

Готовые функции лежат в `cpp_port/src/benchmarks.cpp`.

```powershell
cpp_port\build\synth_benchmark.exe --mode full_domain --json-out benchmark_results.json
cpp_port\build\synth_benchmark.exe --names affine_linear xor_shift_add square
```

Проект разделен так, чтобы ядро синтеза, консольные сценарии и FPGA-часть можно было менять независимо.

### Где Что Лежит

- `cpp_port/` - основная C++20-реализация проекта.
- `cpp_port/include/synth/core.hpp` - публичный интерфейс ядра.
- `cpp_port/src/core.cpp` - основная логика синтеза, проверки и генерации Verilog.
- `cpp_port/src/current_oracle.cpp` - функция, которую пользователь хочет восстановить.
- `cpp_port/src/benchmarks.cpp` - готовые демонстрационные функции.
- `cpp_port/apps/` - CLI-утилиты: поиск, проверка, codegen, benchmark и FPGA-demo.
- `cpp_port/tests/` - автоматические тесты.
- `bench/` - вспомогательные Python-скрипты для сравнительных прогонов.
- `archive/python_reference/` - старая Python-версия, сохраненная как reference.
- `fpga_demo/` - Verilog-обвязка для платы.

Код синтеза находится в одном C++-ядре, а отдельные `.exe` только выбирают сценарий использования.

### Вклад Команды

- Роман Юхарев - C++-ядро, CLI-инструменты, сборка и генерация Verilog.
- Давид Серафимов - FPGA-интеграция, `top.sv`, подключение switches/LEDs/HEX и Quartus scaffold.
- Александр Яковлев - демонстрационные функции, проверка корректности, benchmark-сценарии и оформление учебной части.

