`include "define.svh"

module Comparator_Qint8 (
    input  logic       clk,
    input  logic       rst,
    input  logic [7:0] data_in,
    input  logic       enable,
    input  logic       init_window,
    output logic [7:0] data_out
);
logic [7:0] max_val;

always_ff @(posedge clk or posedge rst) begin
    if (rst || init_window) begin
        max_val <= data_in;
    end else if (enable) begin
        if (data_in > max_val)
            max_val <= data_in;
    end
end

assign data_out = max_val;

endmodule