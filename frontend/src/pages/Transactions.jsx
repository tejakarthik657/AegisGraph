import PageHeader from "../components/layout/PageHeader";
import TransactionTable from "../components/transactions/TransactionTable";
import useTransactions from "../hooks/useTransactions";

export default function Transactions() { const { transactions, loading, error } = useTransactions(); return <><PageHeader title="Transactions" /><p>{loading ? "Loading..." : error || ""}</p><TransactionTable transactions={transactions} /></>; }
