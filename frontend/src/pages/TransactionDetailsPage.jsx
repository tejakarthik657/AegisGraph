import { useParams } from "react-router-dom";
import PageHeader from "../components/layout/PageHeader";
export default function TransactionDetailsPage() { const { transactionId } = useParams(); return <><PageHeader title="Transaction details" /><p>{transactionId}</p></>; }
