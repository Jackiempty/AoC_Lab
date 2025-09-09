`include "./include/define.svh"

module GIN_Bus #(
    parameter NUMS_SLAVE = `NUMS_PE_COL,
    parameter ID_SIZE = `XID_BITS
) (
    input clk,
    input rst,

   // Master I/O
    input [ID_SIZE-1:0] tag,
    input master_valid,
    input [`DATA_BITS-1:0] master_data,
    output logic master_ready, // to GLB

   // Slave I/O
    input [NUMS_SLAVE-1:0] slave_ready, // from pe
    output logic [NUMS_SLAVE-1:0] slave_valid, // to PE
    output logic [`DATA_BITS-1:0] slave_data, // to PE

    // Config
    input set_id,
    input [ID_SIZE-1:0] ID_scan_in,
    output logic [ID_SIZE-1:0] ID_scan_out
);

    logic [ID_SIZE-1:0] id_shift [0:NUMS_SLAVE];
    logic [NUMS_SLAVE-1:0] ready_collect;

    always_comb id_shift[0] = ID_scan_in;
    always_comb ID_scan_out = id_shift[NUMS_SLAVE];

    always_comb master_ready = |ready_collect;

    genvar idx;
    generate
        for (idx = 0; idx < NUMS_SLAVE; idx = idx + 1) begin: MC_ARRAY
            GIN_MulticastController #(
                .ID_SIZE(ID_SIZE)
            ) mc_inst (
                .clk(clk),
                .rst(rst),
                .set_id(set_id),
                .id_in(id_shift[idx]),
                .id(id_shift[idx + 1]),
                .tag(tag),
                .valid_in(master_valid),
                .valid_out(slave_valid[idx]),
                .ready_in(slave_ready[idx]),
                .ready_out(ready_collect[idx])
            );
        end
    endgenerate

    always_comb slave_data = master_data;

endmodule
