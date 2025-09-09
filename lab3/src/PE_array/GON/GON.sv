
`include "src/PE_array/GON/GON_Bus.sv"
`include "src/PE_array/GON/GON_MulticastController.sv"

module GON (
    input clk,
    input rst,

    // Output to GLB
    output logic GON_valid,
    input  GON_ready,
    output logic [`DATA_BITS-1:0] GON_data,

    // Controller interface
    input [`XID_BITS-1:0] tag_X,
    input [`YID_BITS-1:0] tag_Y,

    // Scan chain config
    input set_XID,
    input [`XID_BITS-1:0] XID_scan_in,
    input set_YID,
    input [`YID_BITS-1:0] YID_scan_in,

    // Interface to PE array
    input  [`NUMS_PE_ROW * `NUMS_PE_COL - 1:0] PE_valid,
    output logic [`NUMS_PE_ROW * `NUMS_PE_COL - 1:0] PE_ready,
    input  [`DATA_BITS * `NUMS_PE_ROW * `NUMS_PE_COL - 1:0] PE_data
);


    logic [`XID_BITS-1:0] x_id_chain [0:`NUMS_PE_ROW];

    logic [`YID_BITS-1:0] y_id_chain_out;


    logic [`NUMS_PE_ROW-1:0] x_valid_vec;
    logic [`DATA_BITS * `NUMS_PE_ROW-1:0] x_data_vec;
    logic [`NUMS_PE_ROW-1:0] x_ready_vec;


    always_comb x_id_chain[0] = XID_scan_in;

    genvar row_idx;
    generate
        for (row_idx = 0; row_idx < `NUMS_PE_ROW; row_idx++) begin: GON_X_ROW
            GON_Bus #(
                .NUMS_MASTER(`NUMS_PE_COL),
                .ID_SIZE(`XID_BITS)
            ) x_bus_inst (
                .clk(clk),
                .rst(rst),
                .tag(tag_X),
                .master_valid(PE_valid[`NUMS_PE_COL*(row_idx+1)-1 : `NUMS_PE_COL*row_idx]),
                .master_data(PE_data[`DATA_BITS*`NUMS_PE_COL*(row_idx+1)-1 : `DATA_BITS*`NUMS_PE_COL*row_idx]),
                .master_ready(PE_ready[`NUMS_PE_COL*(row_idx+1)-1 : `NUMS_PE_COL*row_idx]),
                .slave_ready(x_ready_vec[row_idx]),
                .slave_data(x_data_vec[`DATA_BITS*(row_idx+1)-1 : `DATA_BITS*row_idx]),
                .slave_valid(x_valid_vec[row_idx]),
                .set_id(set_XID),
                .ID_scan_in(x_id_chain[row_idx]),
                .ID_scan_out(x_id_chain[row_idx+1])
            );
        end
    endgenerate

    GON_Bus #(
        .NUMS_MASTER(`NUMS_PE_ROW),
        .ID_SIZE(`YID_BITS)
    ) y_bus_inst (
        .clk(clk),
        .rst(rst),
        .tag(tag_Y),
        .master_valid(x_valid_vec),
        .master_data({64'd0, x_data_vec}),
        .master_ready(x_ready_vec),
        .slave_ready(GON_ready),
        .slave_data(GON_data),
        .slave_valid(GON_valid),
        .set_id(set_YID),
        .ID_scan_in(YID_scan_in),
        .ID_scan_out(y_id_chain_out)
    );

endmodule
