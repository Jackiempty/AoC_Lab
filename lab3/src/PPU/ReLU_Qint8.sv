`include "define.svh"

module ReLU_Qint8 (
    input  logic        relu_en,     // ReLU enable
    input  logic signed [31:0] data_in,
    output logic signed [31:0] data_out
);

always_comb begin
    if (relu_en && data_in < 0)
        data_out = 32'sd0;
    else
        data_out = data_in;
end
endmodule