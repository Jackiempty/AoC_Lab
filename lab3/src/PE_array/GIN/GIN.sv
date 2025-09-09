`include "src/PE_array/GIN/GIN_Bus.sv"
`include "src/PE_array/GIN/GIN_MulticastController.sv"

module GIN (
    input clk,
    input rst,

    // Interface to slave SRAM
    input GIN_valid,
    output logic GIN_ready,
    input [`DATA_BITS - 1:0] GIN_data,

    // Controller interface
    input [`XID_BITS - 1:0] tag_X,
    input [`YID_BITS - 1:0] tag_Y,

    // Scan chain config
    input set_XID,
    input [`XID_BITS - 1:0] XID_scan_in,
    input set_YID,
    input [`YID_BITS - 1:0] YID_scan_in,

    // Interface to PE array
    input  [`NUMS_PE_ROW * `NUMS_PE_COL - 1:0] PE_ready,
    output logic [`NUMS_PE_ROW * `NUMS_PE_COL - 1:0] PE_valid,
    output logic [`DATA_BITS - 1:0] PE_data
);


    logic [`XID_BITS-1:0] x_chain [0:`NUMS_PE_ROW];

    logic [`YID_BITS-1:0] y_chain_out;

    logic [`NUMS_PE_ROW-1:0] y_valid_vec;
    logic [`DATA_BITS-1:0]   y_data_bus;
    logic [`NUMS_PE_ROW-1:0] y_ready_vec;


    always_comb x_chain[0] = XID_scan_in;


    genvar row;
    generate
        for (row = 0; row < `NUMS_PE_ROW; row++) begin: X_BUS_GEN
            GIN_Bus #(
                .NUMS_SLAVE(`NUMS_PE_COL),
                .ID_SIZE(`XID_BITS)
            ) x_bus_inst (
                .clk(clk),
                .rst(rst),
                .tag(tag_X),
                .master_valid(y_valid_vec[row]),    
                .master_data(y_data_bus),            
                .master_ready(y_ready_vec[row]),    
                .slave_ready(PE_ready[`NUMS_PE_COL*(row+1)-1 : `NUMS_PE_COL*row]),
                .slave_data(PE_data),              
                .slave_valid(PE_valid[`NUMS_PE_COL*(row+1)-1 : `NUMS_PE_COL*row]),
                .set_id(set_XID),
                .ID_scan_in(x_chain[row]),
                .ID_scan_out(x_chain[row+1])
            );
        end
    endgenerate

    GIN_Bus #(
        .NUMS_SLAVE(`NUMS_PE_ROW),
        .ID_SIZE(`YID_BITS)
    ) y_bus_inst (
        .clk(clk),
        .rst(rst),
        .tag(tag_Y),
        .master_valid(GIN_valid),         
        .master_data(GIN_data),          
        .master_ready(GIN_ready),        
        .slave_ready(y_ready_vec),       
        .slave_data(y_data_bus),         
        .slave_valid(y_valid_vec),       
        .set_id(set_YID),
        .ID_scan_in(YID_scan_in),
        .ID_scan_out(y_chain_out)
    );

endmodule
