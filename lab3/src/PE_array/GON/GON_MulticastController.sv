
`include "./include/define.svh"
module GON_MulticastController #(
    parameter ID_SIZE = `XID_BITS
)(
    input clk,
    input rst,

    // config id
    input set_id,
    input [ID_SIZE - 1:0] id_in,
    output logic [ID_SIZE - 1:0] id,

    // tag
    input [ID_SIZE - 1:0] tag,

    input valid_in, // from pe
    output logic valid_out, // to bus
    input ready_in, // from glb
    output logic ready_out // to pe
);


    always_ff @(posedge clk or posedge rst) begin
        if(rst) id <= '0;
        else id <= set_id ? id_in : id;
    end


    always_comb valid_out = (tag == id && valid_in);
    always_comb ready_out = (tag == id && ready_in);


endmodule
