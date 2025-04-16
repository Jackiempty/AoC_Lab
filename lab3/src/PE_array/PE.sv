`include "define.svh"
module PE (
    input clk,
    input rst,
    input PE_en,
    input [`CONFIG_SIZE-1:0] i_config,
    input [`DATA_BITS-1:0] ifmap,
    input [`DATA_BITS-1:0] filter,
    input [`DATA_BITS-1:0] ipsum,
    input ifmap_valid,
    input filter_valid,
    input ipsum_valid,
    input opsum_ready,
    output logic [`DATA_BITS-1:0] opsum,
    output logic ifmap_ready,
    output logic filter_ready,
    output logic ipsum_ready,
    output logic opsum_valid
);


always_comb begin
    opsum = 32'd0;
    ifmap_ready = 1'd0;
    filter_ready = 1'd0;
    ipsum_ready = 1'd0;
    opsum_valid = 1'd0;
end
endmodule
