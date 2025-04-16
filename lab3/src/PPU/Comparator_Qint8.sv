`include "define.svh"

module Comparator_Qint8 (
    input clk,
    input rst,
    input [`DATA_BITS-1:0] data_in,
    output logic[7:0] data_out
);
always_comb begin
    data_out = 8'd0;
end


endmodule