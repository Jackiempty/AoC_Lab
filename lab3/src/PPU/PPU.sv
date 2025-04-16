`include "src/PPU/post_quant.sv"
`include "src/PPU/Comparator_Qint8.sv"
`include "src/PPU/ReLU_Qint8.sv"
`include "define.svh"

module PPU (
    input clk,
    input rst,
    input [`DATA_BITS-1:0] data_in,
    input [5:0] scaling_factor,
    input maxpool_en,
    input maxpool_init,
    input relu_sel,
    input relu_en,
    output logic[7:0] data_out
);

logic[7:0] output_1;
logic[7:0] output_2;
logic[7:0] output_3;

post_quant post (
    .clk(clk),
    .rst(rst),
    .data_in(data_in),
    .scaling_factor(scaling_factor),
    .data_out(output_1)
);

Comparator_Qint8 comp (
    .clk(clk),
    .rst(rst),
    .data_in(data_in),
    .data_out(output_2)
);

ReLU_Qint8 relu (
    .clk(clk),
    .rst(rst),
    .data_in(data_in),
    .data_out(output_3)
);
 
always_comb begin
    data_out = 8'd0;
end 

endmodule