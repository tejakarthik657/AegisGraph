export default function LoadingSpinner({ label = "Loading..." }) {
  return <div role="status" className="loading-spinner">{label}</div>;
}
