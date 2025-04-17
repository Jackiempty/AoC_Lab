`include "define.svh"

module Comparator_Qint8 (
    input  clk,
    input  rst,
    input  logic [7:0] data_in,
    input  enable,
    input  init_window,
    output logic [7:0] data_out
);
// logic [7:0] count;
logic count = 1;
logic [7:0] max_val;

always_ff @(posedge clk, posedge init_window) begin
    if (init_window && (count == 1)) begin
        // count   <= 8'd0;
        max_val <= 8'd0;
        count <= 0;
    end
    else if (enable) begin
        if (data_in > max_val)
            max_val <= data_in;
        count <= 1;
        // count <= count +1;

    end   
end

always_comb begin
    // max_val = data_in;
    data_out = max_val;
    // data_out = {7'd0, init_window};
end

endmodule