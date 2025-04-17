`include "define.svh"

module Comparator_Qint8 (
    input  logic       clk,
    input  logic       rst,
    input  logic [7:0] data_in,
    input  logic       enable,
    input  logic       init_window,
    output logic [7:0] data_out
);
logic [1:0] count;
logic [7:0] max_val;

always_ff @(posedge clk or posedge rst) begin
    if (rst || init_window) begin
        count   <= 2'd0;
        max_val <= 8'd0;
    end else if (enable) begin
        if (count == 2'd0)
            max_val <= data_in;
        else if (data_in > max_val)
            max_val <= data_in;

        count <= count + 1;

        if (count == 2'd3)
            count <= 2'd0;
    end
end

assign data_out = (count == 2'd3) ? max_val : 8'd0;

endmodule