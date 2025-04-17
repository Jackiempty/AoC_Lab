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
// relu_sel select whether to output the data from post_quant or from max_pool

logic[31:0] output_1;
logic[7:0] output_2;
logic[7:0] output_3;

ReLU_Qint8 relu (
    .relu_en(relu_en),
    .data_in(data_in),
    .data_out(output_1)
);

post_quant post (
    .data_in(output_1),
    .scaling_factor(scaling_factor),
    .data_out(output_2)
);

// maxpooling
Comparator_Qint8 comp (
    .clk         (clk),
    .rst         (rst),
    .data_in     (output_2),
    .enable      (maxpool_en),
    .init_window (maxpool_init),
    .data_out    (output_3)
);
 
always @(posedge clk)
begin
    case(relu_sel)
    1'b0: // post quant
    begin
        data_out = output_2;
    end
    1'b1: // max_pool
    begin
        data_out = output_2;
    end
    endcase
end 

endmodule