`include "define.svh"

module post_quant (
    input [31:0] data_in,
    input [5:0] scaling_factor,
    output logic[7:0] data_out
);
logic signed [31:0] shifted;
logic signed [31:0] added;
logic [7:0]         clamped_temp;

// Step 1: shift (simulate / scale)
always_comb begin
    shifted = data_in >>> scaling_factor; // Arithmetic shift, keep sign
end

// Step 2: add 128, transfer to unsigned
always_comb begin
    added = shifted + 32'sd128;
end

// Step 3: Clamp to 0~255
always_comb begin
    if (added < 0)
        clamped_temp = 8'd0;
    else if (added > 8'd255)
        clamped_temp = 8'd255;
    else
        clamped_temp = added[7:0]; // already within range
end

// Output
always_comb begin
    data_out = clamped_temp[7:0];
end


endmodule