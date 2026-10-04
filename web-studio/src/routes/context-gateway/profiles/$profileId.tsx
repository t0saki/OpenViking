import { createFileRoute } from '@tanstack/react-router'

import { ProfileEditor } from '../-components/profiles-editor'
import { parseProfileEditorSearch } from '../-lib/search'

export const Route = createFileRoute('/context-gateway/profiles/$profileId')({
  validateSearch: parseProfileEditorSearch,
  component: ProfileEditorPage,
})

/** Create (`$profileId` = `new`, `?from=<id>` duplicates) or edit a context profile. */
export function ProfileEditorPage() {
  const { profileId } = Route.useParams()
  const { from } = Route.useSearch()
  return <ProfileEditor profileId={profileId} from={from} />
}
