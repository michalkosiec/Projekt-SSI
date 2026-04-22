# don't think too much about this, it works and it's not the focus of the project,
# just a simple utility to print tables in a nice format, enjoy :)

class TableBuilder:
    def __init__(self, header: list): # creates a builder with the given column names
        self.clen = len(header)
        self.column_widths = [0] * self.clen
        self.rowlists = []
        self.add_row(header)

    def add_row(self, row: list): # adds a row to the table
        if len(row) < self.clen:
            row = row + [""] * (self.clen - len(row))
        
        self.column_widths = [max(self.column_widths[j], len(str(row[j]))) for j in range(self.clen)]
        self.rowlists.append(row)

    def build(self) -> str: # builds the table into a string
        result = [
            self._format_outside_sep(),
            self._format_header(),
            self._format_inside_sep()
            ]
        
        for row in self.rowlists[1:]:
            result.append(self._format_row(row))

        result.append(self._format_outside_sep())
        return "\n".join(result)
    
    # ----- HELPER FORMATTERS -----
    
    def _format_header(self) -> str:
        return self._format_row(self.rowlists[0])
    
    def _format_inside_sep(self) -> str:
        return self._format_row(["-" * self.column_widths[j] for j in range(self.clen)]).replace(" ", "-")
    
    def _format_outside_sep(self) -> str:
        return "+" + ("-" * (len(self._format_header()) - 2)) + "+"
    
    def _format_row(self, row: list) -> str:
        return "| " + " | ".join([str(row[j]).ljust(self.column_widths[j]) for j in range(self.clen)]) + " |"