import { createContext, useContext, useMemo, useState } from "react";

const FilterContext = createContext(null);
export function FilterProvider({ children }) { const [filters, setFilters] = useState({}); const value = useMemo(() => ({ filters, setFilters }), [filters]); return <FilterContext.Provider value={value}>{children}</FilterContext.Provider>; }
export const useFilters = () => useContext(FilterContext);
export default FilterContext;
