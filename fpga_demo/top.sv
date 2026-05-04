module top (
    input  wire        CLOCK_50,
    input  wire [9:0]  SW,
    output wire [9:0]  LEDR,
    output wire [6:0]  HEX0,
    output wire [6:0]  HEX1,
    output wire [6:0]  HEX2,
    output wire [6:0]  HEX3,
    output wire [6:0]  HEX4,
    output wire [6:0]  HEX5
);
    wire [7:0] x;
    wire [7:0] y;

    assign x = SW[7:0];

    generated_function u_generated_function (
        .x(x),
        .y(y)
    );

    assign LEDR[7:0] = y;
    assign LEDR[9:8] = 2'b00;

    hex_to_7seg u_hex0 (
        .value(x[3:0]),
        .segments(HEX0)
    );

    hex_to_7seg u_hex1 (
        .value(x[7:4]),
        .segments(HEX1)
    );

    hex_to_7seg u_hex2 (
        .value(y[3:0]),
        .segments(HEX2)
    );

    hex_to_7seg u_hex3 (
        .value(y[7:4]),
        .segments(HEX3)
    );

    assign HEX4 = 7'b1111111;
    assign HEX5 = 7'b1111111;

endmodule
