export default function ErrorMessage({ message = "Unable to load data." }) {
  return <div role="alert" className="error-message">{message}</div>;
}
