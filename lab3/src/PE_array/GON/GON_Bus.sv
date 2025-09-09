`include "./include/define.svh"
module GON_Bus #(
    parameter NUMS_MASTER = `NUMS_PE_COL,
    parameter ID_SIZE = `XID_BITS
) (
    input clk,
    input rst,
    input [ID_SIZE - 1:0] tag,

    input [NUMS_MASTER - 1:0] master_valid, // from pe
    input [8 * `DATA_BITS - 1:0] master_data, // from pe
    output logic [NUMS_MASTER - 1:0] master_ready, // to pe

    output logic slave_valid, // to glb
    input slave_ready, // from glb
    output logic [`DATA_BITS - 1:0] slave_data, // to glb

    // Config
    input set_id,
    input [ID_SIZE - 1:0] ID_scan_in,
    output logic [ID_SIZE - 1 :0] ID_scan_out
);


    logic [ID_SIZE-1:0] id_shift [0:NUMS_MASTER];
 
    logic [NUMS_MASTER-1:0] valid_to_glb;

    logic [$clog2(NUMS_MASTER)-1:0] selected_idx;
    logic [`DATA_BITS-1:0] master_data_array [0:NUMS_MASTER-1];

    always_comb id_shift[0] = ID_scan_in;
    always_comb ID_scan_out = id_shift[NUMS_MASTER];
    always_comb slave_valid = |valid_to_glb;


    genvar m_idx;
    generate
        for (m_idx = 0; m_idx < NUMS_MASTER; m_idx = m_idx + 1) begin: MC_GEN
            GON_MulticastController #(
                .ID_SIZE(ID_SIZE)
            ) mc_inst (
                .clk(clk),
                .rst(rst),
                .set_id(set_id),
                .id_in(id_shift[m_idx]),
                .id(id_shift[m_idx + 1]),
                .tag(tag),
                .valid_in(master_valid[m_idx]),
                .valid_out(valid_to_glb[m_idx]),
                .ready_in(slave_ready),
                .ready_out(master_ready[m_idx])
            );
        end
    endgenerate


    
    always_comb 
        for (int unpack_idx = 0; unpack_idx < NUMS_MASTER; unpack_idx++) 
            master_data_array[unpack_idx] = master_data[`DATA_BITS * unpack_idx +: `DATA_BITS];



    always_comb begin
        selected_idx = '0;
        for (int k = 0; k < NUMS_MASTER; k++) begin
            if (valid_to_glb[k])
                selected_idx = k[$clog2(NUMS_MASTER)-1:0];
        end
    end


    always_comb slave_data = master_data_array[selected_idx];


endmodule
