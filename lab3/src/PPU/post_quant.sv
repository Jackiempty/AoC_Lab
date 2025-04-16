`include "define.svh"

module post_quant (
    input clk,
    input rst,
    input [`DATA_BITS-1:0] data_in,
    input [5:0] scaling_factor,
    output logic[7:0] data_out
);
always_comb begin
    data_out = 8'd0;
end


endmodule