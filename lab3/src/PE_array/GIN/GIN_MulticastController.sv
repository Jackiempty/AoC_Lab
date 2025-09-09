/* verilator lint_off MULTITOP */
`include "./include/define.svh"
module GIN_MulticastController #(
    parameter ID_SIZE = `XID_BITS
) (
    input clk,
    input rst,

    input set_id,
    input [ID_SIZE - 1:0] id_in,
    output reg [ID_SIZE - 1:0] id,

    input [ID_SIZE - 1:0] tag,
    
    output logic valid_out, // to pe
    input ready_in, // from bus

    output logic ready_out, // to bus
    input valid_in  // from bus
);

    always_ff @(posedge clk or posedge rst) begin
        if(rst) id <= '0;
        else id <= set_id ? id_in : id;
    end
    
    always_comb valid_out = (tag == id && valid_in);

    always_comb ready_out = (tag == id && ready_in);



endmodule
